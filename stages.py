import random, time, traceback
import os.path as osp

import torch
import torch.nn.functional as F
import numpy as np
import yaml

from utils import length_to_mask, maximum_path, log_norm, log_print, get_image
from monotonic_align import mask_from_lens
from losses import magphase_loss

###############################################
# train_first
###############################################


def prepare_batch(batch, device, keys_to_transfer: list["str"] = None) -> tuple:
    """
    Prepares and transfers specified batch elements to the given device.

    This function selects specified elements from a batch and moves them to the specified device
    (e.g., CUDA or CPU). It returns a tuple containing only the requested elements, in the order
    specified by `keys_to_transfer`.

    Parameters
    ----------
    batch : list
        A list containing elements of the batch, where each element corresponds to a predefined key.
        The expected order is:
        0. "waves"
        1. "texts"
        2. "input_lengths"
        3. "ref_texts"
        4. "ref_lengths"
        5. "mels"
        6. "mel_input_length"
        7. "ref_mels"

    device : torch.device or str
        The device to which the selected batch elements should be moved. This can be a PyTorch device object
        (e.g., `torch.device("cuda")`) or a string (`"cuda"`, `"cpu"`).

    keys_to_transfer : list of str, optional
        A list of keys specifying which elements of the batch should be moved to the device.
        If `None`, all elements are transferred. The valid keys are:
        - "waves"
        - "texts"
        - "input_lengths"
        - "ref_texts"
        - "ref_lengths"
        - "mels"
        - "mel_input_length"
        - "ref_mels"

    Returns
    -------
    tuple
        A tuple containing the selected batch elements, moved to the specified device.
        The elements appear in the order they are specified in `keys_to_transfer`.

    Raises
    ------
    ValueError
        If any key in `keys_to_transfer` is not a valid batch key.

    Examples
    --------
    >>> batch = [
    ...     torch.tensor([1, 2, 3]),  # waves
    ...     torch.tensor([4, 5, 6]),  # texts
    ...     torch.tensor([7, 8, 9]),  # input_lengths
    ...     "unused_texts",           # ref_texts (string, just for demonstration)
    ...     "unused_lengths",         # ref_lengths
    ...     torch.tensor([10, 11, 12]), # mels
    ...     torch.tensor([13, 14, 15]), # mel_input_length
    ...     torch.tensor([16, 17, 18])  # ref_mels
    ... ]

    >>> device = torch.device("cuda")

    >>> prepare_batch(batch, device, keys_to_transfer=["waves", "texts", "mels"])
    (tensor([1, 2, 3], device='cuda'), tensor([4, 5, 6], device='cuda'), tensor([10, 11, 12], device='cuda'))
    """
    if keys_to_transfer is None:
        keys_to_transfer = [
            "waves",
            "texts",
            "input_lengths",
            "ref_texts",
            "ref_lengths",
            "mels",
            "mel_input_length",
            "ref_mels",
        ]
    index = {
        "waves": 0,
        "texts": 1,
        "input_lengths": 2,
        "ref_texts": 3,
        "ref_lengths": 4,
        "mels": 5,
        "mel_input_length": 6,
        "ref_mels": 7,
    }
    prepared = tuple()
    for key in keys_to_transfer:
        if key not in index:
            raise ValueError(f"Key {key} not found in batch select from {index.keys()}")
        prepared += (batch[index[key]].to(device),)
    return prepared


def train_first(i, batch, running_loss, iters, train, epoch):
    try:
        waves, texts, input_lengths, mels, mel_input_length = prepare_batch(
            batch,
            train.device,
            ["waves", "texts", "input_lengths", "mels", "mel_input_length"],
        )

        with torch.no_grad():
            mask = length_to_mask(mel_input_length // (2**train.n_down)).to(
                train.device
            )
            text_mask = length_to_mask(input_lengths).to(train.device)

        ppgs, s2s_pred, s2s_attn = train.model.text_aligner(mels, mask, texts)

        s2s_attn = s2s_attn.transpose(-1, -2)
        s2s_attn = s2s_attn[..., 1:]
        s2s_attn = s2s_attn.transpose(-1, -2)

        with torch.no_grad():
            attn_mask = (
                (~mask)
                .unsqueeze(-1)
                .expand(mask.shape[0], mask.shape[1], text_mask.shape[-1])
                .float()
                .transpose(-1, -2)
            )
            attn_mask = (
                attn_mask.float()
                * (~text_mask)
                .unsqueeze(-1)
                .expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1])
                .float()
            )
            attn_mask = attn_mask < 1

        s2s_attn.masked_fill_(attn_mask, 0.0)

        with torch.no_grad():
            mask_ST = mask_from_lens(
                s2s_attn, input_lengths, mel_input_length // (2**train.n_down)
            )
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

        # encode
        t_en = train.model.text_encoder(texts, input_lengths, text_mask)

        # 50% of chance of using monotonic version
        if bool(random.getrandbits(1)):
            asr = t_en @ s2s_attn
        else:
            asr = t_en @ s2s_attn_mono

        en = asr
        gt = mels
        st = mels
        wav = waves

        # clip too short to be used by the style encoder
        if gt.shape[-1] < 40 or (
            gt.shape[-1] < 80 and not train.model_params.skip_downsamples
        ):
            log_print("Skipping batch. TOO SHORT", train.logger)
            return running_loss, iters

        with torch.no_grad():
            real_norm = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
            F0_real, _, _ = train.model.pitch_extractor(gt.unsqueeze(1))

        s = train.model.style_encoder(
            st.unsqueeze(1) if train.multispeaker else gt.unsqueeze(1)
        )

        y_rec, mag_rec, phase_rec = train.model.decoder(en, F0_real, real_norm, s)

        # discriminator loss

        if epoch >= train.TMA_epoch:
            train.optimizer.zero_grad()
            d_loss = train.dl(wav.detach().unsqueeze(1).float(), y_rec.detach()).mean()
            # accelerator.backward(d_loss)
            d_loss.backward()
            train.optimizer.step("msd")
            train.optimizer.step("mpd")
        else:
            d_loss = 0

        # generator loss
        train.optimizer.zero_grad()
        loss_mel = train.stft_loss(y_rec.squeeze(), wav.detach())
        loss_magphase = magphase_loss(mag_rec, phase_rec, wav.detach())
        if epoch >= train.TMA_epoch:  # start TMA training
            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(
                s2s_pred, texts, input_lengths
            ):
                loss_s2s += F.cross_entropy(
                    _s2s_pred[:_text_length], _text_input[:_text_length]
                )
            loss_s2s /= texts.size(0)

            loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

            loss_gen_all = train.gl(wav.detach().unsqueeze(1).float(), y_rec).mean()
            loss_slm = train.wl(wav.detach(), y_rec).mean()

            g_loss = (
                train.loss_params.lambda_mel * loss_mel
                + train.loss_params.lambda_mono * loss_mono
                + train.loss_params.lambda_s2s * loss_s2s
                + train.loss_params.lambda_gen * loss_gen_all
                + train.loss_params.lambda_slm * loss_slm
                + 1 * loss_magphase
            )

        else:
            loss_s2s = 0
            loss_mono = 0
            loss_gen_all = 0
            loss_slm = 0
            g_loss = loss_mel + loss_magphase

        # running_loss += accelerator.gather(loss_mel).mean().item()
        running_loss += loss_mel.item()

        # accelerator.backward(g_loss)
        g_loss.backward()

        train.optimizer.step("text_encoder")
        train.optimizer.step("style_encoder")
        train.optimizer.step("decoder")

        if epoch >= train.TMA_epoch:
            train.optimizer.step("text_aligner")
            train.optimizer.step("pitch_extractor")

        train.iters = train.iters + 1

        if (i + 1) % train.log_interval == 0:  # and accelerator.is_main_process:
            log_print(
                "Epoch [%d/%d], Step [%d/%d], Mel Loss: %.5f, Gen Loss: %.5f, Disc Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f, SLM Loss: %.5f, MP Loss: %.5f"
                % (
                    epoch,
                    train.epochs,
                    i + 1,
                    train.batch_manager.get_step_count(),
                    running_loss / train.log_interval,
                    loss_gen_all,
                    d_loss,
                    loss_mono,
                    loss_s2s,
                    loss_slm,
                    loss_magphase,
                ),
                train.logger,
            )

            train.writer.add_scalar(
                "train/mel_loss", running_loss / train.log_interval, train.iters
            )
            train.writer.add_scalar("train/gen_loss", loss_gen_all, train.iters)
            train.writer.add_scalar("train/d_loss", d_loss, train.iters)
            train.writer.add_scalar("train/mono_loss", loss_mono, train.iters)
            train.writer.add_scalar("train/s2s_loss", loss_s2s, train.iters)
            train.writer.add_scalar("train/slm_loss", loss_slm, train.iters)
            train.writer.add_scalar("train/mp_loss", loss_magphase, train.iters)

            running_loss = 0

            print("Time elasped:", time.time() - train.start_time)
    except Exception as e:
        train.optimizer.zero_grad()
        raise e
    return running_loss, iters


###############################################
# validate_first
###############################################


def validate_first(current_epoch: int, current_step: int, save: bool, train):
    loss_test = 0
    max_len = 1620

    _ = [train.model[key].eval() for key in train.model]

    with torch.no_grad():
        iters_test = 0
        for batch_idx, batch in enumerate(train.val_dataloader):
            train.optimizer.zero_grad()
            waves, texts, input_lengths, mels, mel_input_length = prepare_batch(
                batch,
                train.device,
                ["waves", "texts", "input_lengths", "mels", "mel_input_length"],
            )

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2**train.n_down)).to(
                    train.device
                )
                ppgs, s2s_pred, s2s_attn = train.model.text_aligner(mels, mask, texts)

                s2s_attn = s2s_attn.transpose(-1, -2)
                s2s_attn = s2s_attn[..., 1:]
                s2s_attn = s2s_attn.transpose(-1, -2)

                text_mask = length_to_mask(input_lengths).to(train.device)
                attn_mask = (
                    (~mask)
                    .unsqueeze(-1)
                    .expand(mask.shape[0], mask.shape[1], text_mask.shape[-1])
                    .float()
                    .transpose(-1, -2)
                )
                attn_mask = (
                    attn_mask.float()
                    * (~text_mask)
                    .unsqueeze(-1)
                    .expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1])
                    .float()
                )
                attn_mask = attn_mask < 1
                s2s_attn.masked_fill_(attn_mask, 0.0)

            # encode
            t_en = train.model.text_encoder(texts, input_lengths, text_mask)

            asr = t_en @ s2s_attn

            en = asr
            gt = mels
            wav = waves

            # clip too short to be used by the style encoder
            if gt.shape[-1] < 40 or (
                gt.shape[-1] < 80 and not train.model_params.skip_downsamples
            ):
                log_print("Skipping batch. TOO SHORT", train.logger)
                continue

            F0_real, _, F0 = train.model.pitch_extractor(gt.unsqueeze(1))
            s = train.model.style_encoder(gt.unsqueeze(1))
            real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
            y_rec, _, _ = train.model.decoder(en, F0_real, real_norm, s)

            loss_mel = train.stft_loss(y_rec.squeeze(), wav.detach())

            # loss_test += accelerator.gather(loss_mel).mean().item()
            loss_test += loss_mel.item()
            iters_test += 1
    # if accelerator.is_main_process:
    if True:
        print(
            f"Epochs:{current_epoch} Steps:{current_step} Loss:{loss_test / iters_test} Best_Loss:{train.best_loss}"
        )
        log_print(
            f"Epochs:{current_epoch} Steps:{current_step} Loss:{loss_test / iters_test} Best_Loss:{train.best_loss}",
            train.logger,
        )
        log_print(
            "Validation loss: %.3f" % (loss_test / iters_test) + "\n\n\n\n",
            train.logger,
        )
        print("\n\n\n")
        train.writer.add_scalar("eval/mel_loss", loss_test / iters_test, current_epoch)
        attn_image = get_image(s2s_attn[0].cpu().numpy().squeeze())
        train.writer.add_figure("eval/attn", attn_image, current_epoch)

        with torch.no_grad():
            for bib in range(len(asr)):
                mel_length = int(mel_input_length[bib].item())
                gt = mels[bib, :, :mel_length].unsqueeze(0)
                en = asr[bib, :, : mel_length // 2].unsqueeze(0)

                F0_real, _, _ = train.model.pitch_extractor(gt.unsqueeze(1))
                # F0_real = F0_real.unsqueeze(0)
                s = train.model.style_encoder(gt.unsqueeze(1))
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)

                y_rec, _, _ = train.model.decoder(en, F0_real, real_norm, s)

                train.writer.add_audio(
                    "eval/y" + str(bib),
                    y_rec.cpu().numpy().squeeze(),
                    current_epoch,
                    sample_rate=train.sr,
                )
                if current_epoch == 0:
                    train.writer.add_audio(
                        "gt/y" + str(bib),
                        waves[bib].squeeze(),
                        current_epoch,
                        sample_rate=train.sr,
                    )

                if bib >= 6:
                    break

        if current_epoch % train.saving_epoch == 0 and save and current_step == -1:
            if (loss_test / iters_test) < train.best_loss:
                train.best_loss = loss_test / iters_test
            print("Saving..")
            state = {
                "net": {key: train.model[key].state_dict() for key in train.model},
                "optimizer": train.optimizer.state_dict(),
                "iters": train.iters,
                "val_loss": loss_test / iters_test,
                "epoch": current_epoch,
            }
            save_path = osp.join(train.log_dir, f"epoch_1st_{current_epoch:>05}.pth")
            torch.save(state, save_path)

        if save and current_step != -1:
            if (loss_test / iters_test) < train.best_loss:
                train.best_loss = loss_test / iters_test
            print("Saving..")
            state = {
                "net": {key: train.model[key].state_dict() for key in train.model},
                "optimizer": train.optimizer.state_dict(),
                "iters": train.iters,
                "val_loss": loss_test / iters_test,
                "epoch": current_epoch,
            }
            save_path = osp.join(
                train.log_dir,
                f"epoch_1st_{current_epoch:>05}_step_{current_step:>09}.pth",
            )
            torch.save(state, save_path)
    _ = [train.model[key].train() for key in train.model]  # Restore training mode


###############################################
# train_second
###############################################


def train_second(i, batch, running_loss, iters, train, epoch):
    (
        waves,
        texts,
        input_lengths,
        ref_texts,
        ref_lengths,
        mels,
        mel_input_length,
        ref_mels,
    ) = prepare_batch(
        batch,
        train.device,
        [
            "waves",
            "texts",
            "input_lengths",
            "ref_texts",
            "ref_lengths",
            "mels",
            "mel_input_length",
            "ref_mels",
        ],
    )

    with torch.no_grad():
        mask = length_to_mask(mel_input_length // (2**train.n_down)).to(train.device)
        mel_mask = length_to_mask(mel_input_length).to(train.device)
        text_mask = length_to_mask(input_lengths).to(train.device)

        try:
            _, _, s2s_attn = train.model.text_aligner(mels, mask, texts)
            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)
        except Exception as e:
            print("s2s_attn fail", e)
            return running_loss, iters

        mask_ST = mask_from_lens(
            s2s_attn, input_lengths, mel_input_length // (2**train.n_down)
        )
        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

        # encode
        t_en = train.model.text_encoder(texts, input_lengths, text_mask)
        asr = t_en @ s2s_attn_mono

        d_gt = s2s_attn_mono.sum(axis=-1).detach()

        # compute reference styles
        if train.multispeaker and epoch >= train.diff_epoch:
            ref_ss = train.model.style_encoder(ref_mels.unsqueeze(1))
            ref_sp = train.model.predictor_encoder(ref_mels.unsqueeze(1))
            ref = torch.cat([ref_ss, ref_sp], dim=1)

    # compute the style of the entire utterance
    s_dur = train.model.predictor_encoder(mels.unsqueeze(1))
    gs = train.model.style_encoder(mels.unsqueeze(1))
    s_trg = torch.cat([gs, s_dur], dim=-1).detach()  # ground truth for denoiser

    bert_dur = train.model.bert(texts, attention_mask=(~text_mask).int())
    d_en = train.model.bert_encoder(bert_dur).transpose(-1, -2)

    # denoiser training
    if epoch >= train.diff_epoch:
        num_steps = np.random.randint(3, 5)

        if train.model_params.diffusion.dist.estimate_sigma_data:
            train.model.diffusion.module.diffusion.sigma_data = (
                s_trg.std(axis=-1).mean().item()
            )  # batch-wise std estimation
            train.running_std.append(train.model.diffusion.module.diffusion.sigma_data)

        if train.multispeaker:
            s_preds = train.sampler(
                noise=torch.randn_like(s_trg).unsqueeze(1).to(train.device),
                embedding=bert_dur,
                embedding_scale=1,
                features=ref,  # reference from the same speaker as the embedding
                embedding_mask_proba=0.1,
                num_steps=num_steps,
            ).squeeze(1)
            loss_diff = train.model.diffusion(
                s_trg.unsqueeze(1), embedding=bert_dur, features=ref
            ).mean()  # EDM loss
            loss_sty = F.l1_loss(s_preds, s_trg.detach())  # style reconstruction loss
        else:
            s_preds = train.sampler(
                noise=torch.randn_like(s_trg).unsqueeze(1).to(train.device),
                embedding=bert_dur,
                embedding_scale=1,
                embedding_mask_proba=0.1,
                num_steps=num_steps,
            ).squeeze(1)
            loss_diff = train.model.diffusion.module.diffusion(
                s_trg.unsqueeze(1), embedding=bert_dur
            ).mean()  # EDM loss
            loss_sty = F.l1_loss(s_preds, s_trg.detach())  # style reconstruction loss
    else:
        loss_sty = 0
        loss_diff = 0

    d, p_en = train.model.predictor(
        d_en, s_dur, input_lengths, s2s_attn_mono, text_mask
    )

    wav = waves

    if mels.shape[-1] < 40 or (
        mels.shape[-1] < 80 and not train.model_params.skip_downsamples
    ):
        log_print("Skipping batch. TOO SHORT", train.logger)
        return running_loss, iters

    with torch.no_grad():
        F0_real, _, _ = train.model.pitch_extractor(mels.unsqueeze(1))
        N_real = log_norm(mels.unsqueeze(1)).squeeze(1)

        wav = wav.unsqueeze(1)
        y_rec_gt = wav

        if epoch >= train.joint_epoch:
            # ground truth from reconstruction
            y_rec_gt_pred, _, _ = train.model.decoder(asr, F0_real, N_real, gs)

    F0_fake, N_fake = train.model.predictor.F0Ntrain(p_en, s_dur)

    y_rec, mag_rec, phase_rec = train.model.decoder(asr, F0_fake, N_fake, gs)
    loss_magphase = magphase_loss(mag_rec, phase_rec, wav.squeeze(1).detach())

    loss_F0_rec = (F.smooth_l1_loss(F0_real, F0_fake)) / 10
    loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

    if train.start_ds:
        train.optimizer.zero_grad()
        d_loss = train.dl(wav.detach(), y_rec.detach()).mean()
        d_loss.backward()
        train.optimizer.step("msd")
        train.optimizer.step("mpd")
    else:
        d_loss = 0

    # generator loss
    train.optimizer.zero_grad()

    loss_mel = train.stft_loss(y_rec, wav)
    if train.start_ds:
        loss_gen_all = train.gl(wav, y_rec).mean()
    else:
        loss_gen_all = 0
    loss_lm = train.wl(wav.detach().squeeze(1), y_rec.squeeze(1)).mean()

    loss_ce = 0
    loss_dur = 0
    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
        _s2s_pred = _s2s_pred[:_text_length, :]
        _text_input = _text_input[:_text_length].long()
        _s2s_trg = torch.zeros_like(_s2s_pred)
        for p in range(_s2s_trg.shape[0]):
            _s2s_trg[p, : _text_input[p]] = 1
        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

        loss_dur += F.l1_loss(
            _dur_pred[1 : _text_length - 1], _text_input[1 : _text_length - 1]
        )
        loss_ce += F.binary_cross_entropy_with_logits(
            _s2s_pred.flatten(), _s2s_trg.flatten()
        )

    loss_ce /= texts.size(0)
    loss_dur /= texts.size(0)

    g_loss = (
        train.loss_params.lambda_mel * loss_mel
        + train.loss_params.lambda_F0 * loss_F0_rec
        + train.loss_params.lambda_ce * loss_ce
        + train.loss_params.lambda_norm * loss_norm_rec
        + train.loss_params.lambda_dur * loss_dur
        + train.loss_params.lambda_gen * loss_gen_all
        + train.loss_params.lambda_slm * loss_lm
        + train.loss_params.lambda_sty * loss_sty
        + train.loss_params.lambda_diff * loss_diff
        + 1 * loss_magphase
    )

    running_loss += loss_mel.item()
    g_loss.backward()
    if torch.isnan(g_loss):
        from IPython.core.debugger import set_trace

        set_trace()

    train.optimizer.step("bert_encoder")
    train.optimizer.step("bert")
    train.optimizer.step("predictor")
    train.optimizer.step("predictor_encoder")

    if epoch >= train.diff_epoch:
        train.optimizer.step("diffusion")

    if epoch >= train.joint_epoch or train.early_joint:
        train.optimizer.step("style_encoder")
        train.optimizer.step("decoder")

    if epoch >= train.joint_epoch:
        # randomly pick whether to use in-distribution text
        if np.random.rand() < 0.5:
            use_ind = True
        else:
            use_ind = False

        if use_ind:
            ref_lengths = input_lengths
            ref_texts = texts

        slm_out = train.slmadv(
            i,
            y_rec_gt,
            y_rec_gt_pred,
            waves,
            mel_input_length,
            ref_texts,
            ref_lengths,
            use_ind,
            s_trg.detach(),
            ref if train.multispeaker else None,
        )

        if slm_out is None:
            print("slm_out none")
            return running_loss, iters

        d_loss_slm, loss_gen_lm, y_pred = slm_out

        # SLM generator loss
        train.optimizer.zero_grad()
        loss_gen_lm.backward()

        # compute the gradient norm
        total_norm = {}
        for key in train.model.keys():
            total_norm[key] = 0
            parameters = [
                p
                for p in train.model[key].parameters()
                if p.grad is not None and p.requires_grad
            ]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm[key] += param_norm.item() ** 2
            total_norm[key] = total_norm[key] ** 0.5

        # gradient scaling
        if total_norm["predictor"] > train.slmadv_params.thresh:
            for key in train.model.keys():
                for p in train.model[key].parameters():
                    if p.grad is not None:
                        p.grad *= 1 / total_norm["predictor"]

        for p in train.model.predictor.duration_proj.parameters():
            if p.grad is not None:
                p.grad *= train.slmadv_params.scale

        for p in train.model.predictor.lstm.parameters():
            if p.grad is not None:
                p.grad *= train.slmadv_params.scale

        for p in train.model.diffusion.parameters():
            if p.grad is not None:
                p.grad *= train.slmadv_params.scale

        train.optimizer.step("bert_encoder")
        train.optimizer.step("bert")
        train.optimizer.step("predictor")
        train.optimizer.step("diffusion")

        # SLM discriminator loss
        if d_loss_slm != 0:
            train.optimizer.zero_grad()
            d_loss_slm.backward(retain_graph=True)
            train.optimizer.step("wd")

    else:
        d_loss_slm, loss_gen_lm = 0, 0

    train.iters = train.iters + 1

    if (i + 1) % train.log_interval == 0:
        train.logger.info(
            "Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, LM Loss: %.5f, Gen Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f, DiscLM Loss: %.5f, GenLM Loss: %.5f, MP Loss: %.5f"
            % (
                epoch,
                train.epochs,
                i + 1,
                train.batch_manager.get_step_count(),
                running_loss / train.log_interval,
                d_loss,
                loss_dur,
                loss_ce,
                loss_norm_rec,
                loss_F0_rec,
                loss_lm,
                loss_gen_all,
                loss_sty,
                loss_diff,
                d_loss_slm,
                loss_gen_lm,
                loss_magphase,
            )
        )

        train.writer.add_scalar(
            "train/mel_loss", running_loss / train.log_interval, train.iters
        )
        train.writer.add_scalar("train/gen_loss", loss_gen_all, train.iters)
        train.writer.add_scalar("train/d_loss", d_loss, train.iters)
        train.writer.add_scalar("train/ce_loss", loss_ce, train.iters)
        train.writer.add_scalar("train/dur_loss", loss_dur, train.iters)
        train.writer.add_scalar("train/slm_loss", loss_lm, train.iters)
        train.writer.add_scalar("train/norm_loss", loss_norm_rec, train.iters)
        train.writer.add_scalar("train/F0_loss", loss_F0_rec, train.iters)
        train.writer.add_scalar("train/sty_loss", loss_sty, train.iters)
        train.writer.add_scalar("train/diff_loss", loss_diff, train.iters)
        train.writer.add_scalar("train/d_loss_slm", d_loss_slm, train.iters)
        train.writer.add_scalar("train/gen_loss_slm", loss_gen_lm, train.iters)

        running_loss = 0

        print("Time elasped:", time.time() - train.start_time)
    # optimizer.scheduler()
    return running_loss, iters


###############################################
# validate_second
###############################################


def validate_second(current_epoch: int, current_step: int, save: bool, train):
    loss_test = 0
    loss_align = 0
    loss_f = 0
    _ = [train.model[key].eval() for key in train.model]

    with torch.no_grad():
        iters_test = 0
        for batch_idx, batch in enumerate(train.val_dataloader):
            train.optimizer.zero_grad()

            try:
                waves, texts, input_lengths, mels, mel_input_length, ref_mels = (
                    prepare_batch(
                        batch,
                        train.device,
                        [
                            "waves",
                            "texts",
                            "input_lengths",
                            "mels",
                            "mel_input_length",
                            "ref_mels",
                        ],
                    )
                )

                with torch.no_grad():
                    mask = length_to_mask(mel_input_length // (2**train.n_down)).to(
                        train.device
                    )
                    text_mask = length_to_mask(input_lengths).to(texts.device)

                    _, _, s2s_attn = train.model.text_aligner(mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)

                    mask_ST = mask_from_lens(
                        s2s_attn, input_lengths, mel_input_length // (2**train.n_down)
                    )
                    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                    # encode
                    t_en = train.model.text_encoder(texts, input_lengths, text_mask)
                    asr = t_en @ s2s_attn_mono

                    d_gt = s2s_attn_mono.sum(axis=-1).detach()

                # clip too short to be used by the style encoder
                if mels.shape[-1] < 40 or (
                    mels.shape[-1] < 80 and not train.model_params.skip_downsamples
                ):
                    log_print("Skipping batch. TOO SHORT", train.logger)
                    continue

                s = train.model.predictor_encoder(mels.unsqueeze(1))
                gs = train.model.style_encoder(mels.unsqueeze(1))
                s_trg = torch.cat([s, gs], dim=-1).detach()

                bert_dur = train.model.bert(texts, attention_mask=(~text_mask).int())
                d_en = train.model.bert_encoder(bert_dur).transpose(-1, -2)
                d, p = train.model.predictor(
                    d_en, s, input_lengths, s2s_attn_mono, text_mask
                )

                wav = waves
                en = asr
                p_en = p
                gt = mels.detach()

                F0_fake, N_fake = train.model.predictor.F0Ntrain(p_en, s)

                loss_dur = 0
                for _s2s_pred, _text_input, _text_length in zip(
                    d, (d_gt), input_lengths
                ):
                    _s2s_pred = _s2s_pred[:_text_length, :]
                    _text_input = _text_input[:_text_length].long()
                    _s2s_trg = torch.zeros_like(_s2s_pred)
                    for bib in range(_s2s_trg.shape[0]):
                        _s2s_trg[bib, : _text_input[bib]] = 1
                    _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                    loss_dur += F.l1_loss(
                        _dur_pred[1 : _text_length - 1],
                        _text_input[1 : _text_length - 1],
                    )

                loss_dur /= texts.size(0)

                y_rec, _, _ = train.model.decoder(en, F0_fake, N_fake, gs)
                loss_mel = train.stft_loss(y_rec.squeeze(1), wav.detach())

                F0_real, _, F0 = train.model.pitch_extractor(gt.unsqueeze(1))

                loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                loss_test += (loss_mel).mean()
                loss_align += (loss_dur).mean()
                loss_f += (loss_F0).mean()

                iters_test += 1
            except Exception as e:
                print(f"run into exception", e)
                traceback.print_exc()
                continue

    print("Epochs:", current_epoch)
    train.logger.info(
        "Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f"
        % (loss_test / iters_test, loss_align / iters_test, loss_f / iters_test)
        + "\n\n\n"
    )
    print("\n\n\n")
    train.writer.add_scalar("eval/mel_loss", loss_test / iters_test, current_epoch)
    train.writer.add_scalar("eval/dur_loss", loss_align / iters_test, current_epoch)
    train.writer.add_scalar("eval/F0_loss", loss_f / iters_test, current_epoch)
    # if epoch < joint_epoch:
    if False:
        # generating reconstruction examples with GT duration

        with torch.no_grad():
            for bib in range(len(asr)):
                mel_length = int(mel_input_length[bib].item())
                gt = mels[bib, :, :mel_length].unsqueeze(0)
                en = asr[bib, :, : mel_length // 2].unsqueeze(0)

                F0_real, _, _ = train.model.pitch_extractor(gt.unsqueeze(1))
                F0_real = F0_real.unsqueeze(0)
                s = train.model.style_encoder(gt.unsqueeze(1))
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)

                y_rec, _, _ = train.model.decoder(en, F0_real, real_norm, s)

                writer.add_audio(
                    "eval/y" + str(bib),
                    y_rec.cpu().numpy().squeeze(1),
                    epoch,
                    sample_rate=sr,
                )

                s_dur = train.model.predictor_encoder(gt.unsqueeze(1))
                p_en = p[bib, :, : mel_length // 2].unsqueeze(0)

                F0_fake, N_fake = train.model.predictor.F0Ntrain(p_en, s_dur)

                y_pred, _, _ = train.model.decoder(en, F0_fake, N_fake, s)

                writer.add_audio(
                    "pred/y" + str(bib),
                    y_pred.cpu().numpy().squeeze(1),
                    epoch,
                    sample_rate=sr,
                )

                if epoch == 1:
                    writer.add_audio(
                        "gt/y" + str(bib),
                        waves[bib].squeeze(1),
                        epoch,
                        sample_rate=sr,
                    )

                if bib >= 5:
                    break
    elif False:
        # generating sampled speech from text directly
        with torch.no_grad():
            # compute reference styles
            if multispeaker and epoch >= diff_epoch:
                ref_ss = train.model.style_encoder(ref_mels.unsqueeze(1))
                ref_sp = train.model.predictor_encoder(ref_mels.unsqueeze(1))
                ref_s = torch.cat([ref_ss, ref_sp], dim=1)

            for bib in range(len(d_en)):
                if multispeaker:
                    s_pred = sampler(
                        noise=torch.randn((1, 256)).unsqueeze(1).to(texts.device),
                        embedding=bert_dur[bib].unsqueeze(0),
                        embedding_scale=1,
                        features=ref_s[bib].unsqueeze(
                            0
                        ),  # reference from the same speaker as the embedding
                        num_steps=5,
                    ).squeeze(1)
                else:
                    s_pred = sampler(
                        noise=torch.randn((1, 256)).unsqueeze(1).to(texts.device),
                        embedding=bert_dur[bib].unsqueeze(0),
                        embedding_scale=1,
                        num_steps=5,
                    ).squeeze(1)

                s = s_pred[:, 128:]
                ref = s_pred[:, :128]

                d = train.model.predictor.text_encoder(
                    d_en[bib, :, : input_lengths[bib]].unsqueeze(0),
                    s,
                    input_lengths[bib, ...].unsqueeze(0),
                    text_mask[bib, : input_lengths[bib]].unsqueeze(0),
                )

                x, _ = train.model.predictor.lstm(d)
                duration = train.model.predictor.duration_proj(x)

                duration = torch.sigmoid(duration).sum(axis=-1)
                pred_dur = torch.round(duration.squeeze(1)).clamp(min=1)

                pred_dur[-1] += 5

                pred_aln_trg = torch.zeros(input_lengths[bib], int(pred_dur.sum().data))
                c_frame = 0
                for i in range(pred_aln_trg.size(0)):
                    pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                    c_frame += int(pred_dur[i].data)

                # encode prosody
                en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(texts.device)
                F0_pred, N_pred = train.model.predictor.F0Ntrain(en, s)
                out, _, _ = train.model.decoder(
                    (
                        t_en[bib, :, : input_lengths[bib]].unsqueeze(0)
                        @ pred_aln_trg.unsqueeze(0).to(texts.device)
                    ),
                    F0_pred,
                    N_pred,
                    ref.squeeze(1).unsqueeze(0),
                )

                writer.add_audio(
                    "pred/y" + str(bib),
                    out.cpu().numpy().squeeze(1),
                    current_epoch,
                    sample_rate=sr,
                )

                if bib >= 5:
                    break

    if current_epoch % train.saving_epoch == 0:
        if (loss_test / iters_test) < train.best_loss:
            best_loss = loss_test / iters_test
        print("Saving..")
        state = {
            "net": {key: train.model[key].state_dict() for key in train.model},
            "optimizer": train.optimizer.state_dict(),
            "iters": train.iters,
            "val_loss": loss_test / iters_test,
            "epoch": current_epoch,
        }
        save_path = osp.join(train.log_dir, "epoch_2nd_%05d.pth" % current_epoch)
        torch.save(state, save_path)

        # if estimate sigma, save the estimated simga
        if train.model_params.diffusion.dist.estimate_sigma_data:
            train.config["model_params"]["diffusion"]["dist"]["sigma_data"] = float(
                np.mean(train.running_std)
            )

            with open(
                osp.join(train.log_dir, osp.basename(train.config_path)), "w"
            ) as outfile:
                yaml.dump(train.config, outfile, default_flow_style=True)
