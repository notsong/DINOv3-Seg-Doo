# Progressive finetuning

User approved direct implementation and GitHub upload. Keep existing isolated branch and publish it after verification; no long training.

- Add config: freeze_epochs=10, partial_unfreeze_layers=4, full_unfreeze_epoch=30 (-1 disables full unfreeze), layer_lr_decay=0.8, finetune_warmup_epochs=3. Existing head learning_rate=1e-4 and backbone unfreeze_lr=1e-5 remain.
- Build all AdamW groups once, including frozen parameters. Partition embeddings, each transformer layer, other backbone parameters and decoder/detail parameters without duplication. Earlier layers get exponentially lower base LR.
- Determine trainability from epoch, never from an equality-only event. Apply local warmup/cosine per activation time; use proportional LR floors so layer ordering survives decay.
- Preserve optimizer momentum across stages. Set frozen backbone modules to eval after model.train().
- Save policy, optimizer, scaler, best metrics and RNG in checkpoints. Restore policy-compatible checkpoints; fail clearly on older incompatible optimizer state instead of silently restarting training.
- Test stage boundaries, masks, learning-rate ratios, preserved momentum, resume before/after boundaries and invalid configurations. Add real backbone stage-gradient smoke checks.
- Run all tests, smoke, compile, diff checks, inspect staged content, commit and push authorized code to origin work branch. Verify remote SHA.
