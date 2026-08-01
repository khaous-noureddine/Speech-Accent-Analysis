# configfile: "experiments/stage2-dtw/fold_00/configs/stage2-dtw.yaml"

if "supervised_contrastive_training" in config:
    rule stage2_supcon_dtw_l2cv:
        input:
            script  = "stage2/supcon_train_meanpool.py",
            parquet = config["supervised_contrastive_training_l2cv"]["data"]["parquet_path"],
        output:
            checkpoint = (
                config["supervised_contrastive_training_l2cv"]["training"]["checkpoint_dir"]
                + "/checkpoint_final.pt"
            ),
        params:
            # data
            sample_rate     = config["supervised_contrastive_training_l2cv"]["data"].get("sample_rate", 16000),
            max_audio_len_s = config["supervised_contrastive_training_l2cv"]["data"].get("max_audio_len_s", 10.0),
            label_col       = config["supervised_contrastive_training_l2cv"]["data"].get("label_col", "prompt_id"),
            num_workers     = config["supervised_contrastive_training_l2cv"]["data"].get("num_workers", 2),
            train_split     = config["supervised_contrastive_training_l2cv"]["data"].get("train_split", "train"),
            dev_split       = config["supervised_contrastive_training_l2cv"]["data"].get("dev_split", "dev"),
            validate_audio  = str(config["supervised_contrastive_training_l2cv"]["data"].get("validate_audio", True)).lower(),

            # sampler
            k_utterances = config["supervised_contrastive_training_l2cv"]["sampler"].get("k_utterances", 6),
            s_speakers   = config["supervised_contrastive_training_l2cv"]["sampler"].get("s_speakers", 4),
            n_batches    = config["supervised_contrastive_training_l2cv"]["sampler"].get("n_batches", 100),
            seed         = config["supervised_contrastive_training_l2cv"]["sampler"].get("seed", 42),

            # model
            model_name            = config["supervised_contrastive_training_l2cv"]["model"].get("model_name", "facebook/wav2vec2-large-xlsr-53"),
            proj_hidden_dim       = config["supervised_contrastive_training_l2cv"]["model"].get("proj_hidden_dim", 512),
            proj_out_dim          = config["supervised_contrastive_training_l2cv"]["model"].get("proj_out_dim", 256),
            vocab_size            = config["supervised_contrastive_training_l2cv"]["model"].get("vocab_size", 32),
            min_frozen_layer      = config["supervised_contrastive_training_l2cv"]["model"].get("min_frozen_layer", 0),
            max_frozen_layer      = config["supervised_contrastive_training_l2cv"]["model"].get("max_frozen_layer", 0),
            ctc_lambda            = config["supervised_contrastive_training_l2cv"]["model"].get("ctc_lambda", 0.3),
            temperature           = config["supervised_contrastive_training_l2cv"]["model"].get("temperature", 0.1),
            dtw_gamma             = config["supervised_contrastive_training_l2cv"]["model"].get("dtw_gamma", 0.1),
            dtw_max_frames        = config["supervised_contrastive_training_l2cv"]["model"].get("dtw_max_frames", 80),
            dtw_downsample_stride = config["supervised_contrastive_training_l2cv"]["model"].get("dtw_downsample_stride", 4),
            dtw_pair_chunk        = config["supervised_contrastive_training_l2cv"]["model"].get("dtw_pair_chunk", 4096),

            # training
            experiment_name     = config["experiment"]["name"],
            epochs              = config["supervised_contrastive_training_l2cv"]["training"].get("epochs", 30),
            lr                  = config["supervised_contrastive_training_l2cv"]["training"].get("learning_rate", 1e-5),
            weight_decay        = config["supervised_contrastive_training_l2cv"]["training"].get("weight_decay", 1e-4),
            warmup_steps        = config["supervised_contrastive_training_l2cv"]["training"].get("warmup_steps", 100),
            grad_clip           = config["supervised_contrastive_training_l2cv"]["training"].get("grad_clip", 1.0),
            use_ctc             = str(config["supervised_contrastive_training_l2cv"]["training"].get("use_ctc", True)).lower(),
            tokenizer           = config["supervised_contrastive_training_l2cv"]["training"].get("tokenizer", "facebook/wav2vec2-large-960h"),
            device              = config["supervised_contrastive_training_l2cv"]["training"].get("device", "cuda"),
            use_mixed_precision = str(config["supervised_contrastive_training_l2cv"]["training"].get("use_mixed_precision", True)).lower(),
            checkpoint_dir      = config["supervised_contrastive_training_l2cv"]["training"]["checkpoint_dir"],
            tensorboard_dir     = config["supervised_contrastive_training_l2cv"]["training"]["tensorboard_dir"],
            save_every_n_epochs = config["supervised_contrastive_training_l2cv"]["training"].get("save_every_n_epochs", 1),
            verbose_timing      = str(config["supervised_contrastive_training_l2cv"]["training"].get("verbose_timing", False)).lower(),
            log_every_n_batches = config["supervised_contrastive_training_l2cv"]["training"].get("log_every_n_batches", 1),

            # evaluation
            eval_every_n_epochs = config["supervised_contrastive_training_l2cv"]["evaluation"].get("eval_every_n_epochs", 1),
            eval_batch_size     = config["supervised_contrastive_training_l2cv"]["evaluation"].get("eval_batch_size", 8),
            eval_max_batches    = config["supervised_contrastive_training_l2cv"]["evaluation"].get("eval_max_batches", 4),
            eval_n_neg_samples  = config["supervised_contrastive_training_l2cv"]["evaluation"].get("eval_n_neg_samples", 5000),
            eval_output_dir     = config["supervised_contrastive_training_l2cv"]["evaluation"].get(
                "eval_output_dir",
                "experiments/stage2-dtw/fold_00/eval/stage_2_representation"
            ),
            retrieval_ks        = ",".join(
                map(str, config["supervised_contrastive_training_l2cv"]["evaluation"].get("retrieval_ks", [1, 5, 10]))
            ),
            best_metric         = config["supervised_contrastive_training_l2cv"]["evaluation"].get(
                "best_metric",
                "alignment_ratio_backbone"
            ),

            # slurm
            partition = config.get("slurm", {}).get("partition", "GPU-H200"),
            account   = config.get("slurm", {}).get("account", "efl"),
            gres      = config.get("slurm", {}).get("gres", "gpu:1"),
            cpus      = config.get("slurm", {}).get("cpus_per_task", 4),
            mem       = config.get("slurm", {}).get("mem", "64G"),
            time      = config.get("slurm", {}).get("time", "2-00:00:00"),
            eval_k_utterances = config["supervised_contrastive_training_l2cv"]["evaluation"].get("eval_k_utterances", 8),
            eval_s_speakers   = config["supervised_contrastive_training_l2cv"]["evaluation"].get("eval_s_speakers", 3),
            eval_n_batches    = config["supervised_contrastive_training_l2cv"]["evaluation"].get("eval_n_batches", 20),
        shell:
            r"""
            mkdir -p {params.checkpoint_dir}
            mkdir -p {params.tensorboard_dir}
            mkdir -p {params.eval_output_dir}


            python {input.script} \
                --parquet_path {input.parquet} \
                --sample_rate {params.sample_rate} \
                --max_audio_len_s {params.max_audio_len_s} \
                --label_col {params.label_col} \
                --num_workers {params.num_workers} \
                --train_split {params.train_split} \
                --dev_split {params.dev_split} \
                --validate_audio {params.validate_audio} \
                --k_utterances {params.k_utterances} \
                --s_speakers {params.s_speakers} \
                --n_batches {params.n_batches} \
                --seed {params.seed} \
                --model_name {params.model_name} \
                --proj_hidden_dim {params.proj_hidden_dim} \
                --proj_out_dim {params.proj_out_dim} \
                --vocab_size {params.vocab_size} \
                --min_frozen_layer {params.min_frozen_layer} \
                --max_frozen_layer {params.max_frozen_layer} \
                --ctc_lambda {params.ctc_lambda} \
                --temperature {params.temperature} \
                --dtw_gamma {params.dtw_gamma} \
                --dtw_max_frames {params.dtw_max_frames} \
                --dtw_downsample_stride {params.dtw_downsample_stride} \
                --dtw_pair_chunk {params.dtw_pair_chunk} \
                --epochs {params.epochs} \
                --lr {params.lr} \
                --weight_decay {params.weight_decay} \
                --warmup_steps {params.warmup_steps} \
                --grad_clip {params.grad_clip} \
                --use_ctc {params.use_ctc} \
                --tokenizer {params.tokenizer} \
                --device {params.device} \
                --use_mixed_precision {params.use_mixed_precision} \
                --save_dir {params.checkpoint_dir} \
                --tensorboard_dir {params.tensorboard_dir} \
                --save_every_n_epochs {params.save_every_n_epochs} \
                --eval_every_n_epochs {params.eval_every_n_epochs} \
                --eval_batch_size {params.eval_batch_size} \
                --eval_max_batches {params.eval_max_batches} \
                --eval_n_neg_samples {params.eval_n_neg_samples} \
                --eval_output_dir {params.eval_output_dir} \
                --retrieval_ks "{params.retrieval_ks}" \
                --best_metric {params.best_metric} \
                --verbose_timing {params.verbose_timing} \
                --log_every_n_batches {params.log_every_n_batches} \
                --eval_k_utterances {params.eval_k_utterances} \
                --eval_s_speakers {params.eval_s_speakers} \
                --eval_n_batches {params.eval_n_batches}
            """






# --------------------------------------- #
# ASR Finetuning on LibriSpeech or AESRC  #
# --------------------------------------- #

if "asr_finetuning" in config:
    _EXP_NAME = config["experiment"]["name"]
    _EXP_DIR  = f"experiments/{_EXP_NAME}"

    rule asr_finetuning:
        input:
            script        = "stage3/asr_train.py",
            train_parquet = config["asr_finetuning"]["data"]["train_parquet"],
            eval_parquet  = config["asr_finetuning"]["data"]["eval_parquet"],
        output:
            checkpoint = f"{_EXP_DIR}/{config['asr_finetuning']['training']['output_dir']}/checkpoint_final.pt"
        params:
            dataset           = config["asr_finetuning"]["data"]["dataset"],
            max_duration_s    = config["asr_finetuning"]["data"]["max_duration_s"],
            num_workers       = config["asr_finetuning"]["data"]["num_workers"],
            stage2_checkpoint = config["asr_finetuning"]["model"]["stage2_checkpoint"],
            model_name        = config["asr_finetuning"]["model"]["model_name"],
            epochs            = config["asr_finetuning"]["training"]["epochs"],
            max_steps         = config["asr_finetuning"]["training"]["max_steps"],
            batch_size        = config["asr_finetuning"]["training"]["batch_size"],
            backbone_lr       = config["asr_finetuning"]["training"]["backbone_lr"],
            head_lr           = config["asr_finetuning"]["training"]["head_lr"],
            weight_decay      = config["asr_finetuning"]["training"]["weight_decay"],
            warmup_ratio      = config["asr_finetuning"]["training"]["warmup_ratio"],
            grad_clip         = config["asr_finetuning"]["training"]["grad_clip"],
            device            = config["asr_finetuning"]["training"]["device"],
            log_every         = config["asr_finetuning"]["training"]["log_every"],
            save_every        = config["asr_finetuning"]["training"]["save_every"],
            save_every_steps  = config["asr_finetuning"]["training"]["save_every_steps"],
            checkpoint_dir    = f"{_EXP_DIR}/{config['asr_finetuning']['training']['output_dir']}",
            tensorboard_dir   = f"{_EXP_DIR}/{config['asr_finetuning']['training']['tensorboard_dir']}",
            logging_dir       = f"{_EXP_DIR}/{config['asr_finetuning']['training']['logging_dir']}",
            eval_every        = config["asr_finetuning"]["evaluation"]["eval_every"],
            eval_every_steps  = config["asr_finetuning"]["evaluation"]["eval_every_steps"],
        shell:
            """
            export LD_PRELOAD={workflow.basedir}/.pixi/envs/default/lib/libstdc++.so.6
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

            STAGE2_ARG=""
            if [ -n "{params.stage2_checkpoint}" ]; then
                STAGE2_ARG="--stage2_checkpoint {params.stage2_checkpoint}"
            fi

            srun -p GPU-H200 \
                --job-name=asr_finetuning \
                --account=efl \
                --gres=gpu:1 \
                --cpus-per-task=4 \
                --mem=64G \
                --time=2-00:00:00 \
                python {input.script} \
                    --dataset         {params.dataset} \
                    --model_name      {params.model_name} \
                    $STAGE2_ARG \
                    --train_parquet   {input.train_parquet} \
                    --eval_parquet    {input.eval_parquet} \
                    --max_duration_s  {params.max_duration_s} \
                    --num_workers     {params.num_workers} \
                    --epochs          {params.epochs} \
                    --max_steps       {params.max_steps} \
                    --batch_size      {params.batch_size} \
                    --backbone_lr     {params.backbone_lr} \
                    --head_lr         {params.head_lr} \
                    --weight_decay    {params.weight_decay} \
                    --warmup_ratio    {params.warmup_ratio} \
                    --grad_clip       {params.grad_clip} \
                    --log_every       {params.log_every} \
                    --save_every      {params.save_every} \
                    --save_every_steps {params.save_every_steps} \
                    --eval_every      {params.eval_every} \
                    --eval_every_steps {params.eval_every_steps} \
                    --output_dir      {params.checkpoint_dir} \
                    --device          {params.device} \
                    --tensorboard_dir {params.tensorboard_dir} \
                    --logging_dir     {params.logging_dir}
            """





# --------------------------------------#
# Greedy Evaluation                     #
# --------------------------------------#
if "evaluation" in config:
    _EXP_NAME = config["experiment"]["name"]
    _EXP_DIR  = f"experiments/{_EXP_NAME}"

    _eval_cfg   = config["evaluation"]
    _output_dir = f"{_EXP_DIR}/{_eval_cfg['output_dir']}"

    _all_csvs = [
        f"{_output_dir}/transcriptions/{d['name']}/{m.get('label', m.get('model', m.get('name', ''))).replace('/', '_')}.csv"
        for m in _eval_cfg["models"]
        for d in _eval_cfg["datasets"]
    ]

    rule eval_transcribe:
        input:
            script = "evaluation/transcribe.py",
            #model_ready = f"{_EXP_DIR}/{config['asr_finetuning']['training']['output_dir']}/checkpoint_final.pt"
        output:
            csvs = _all_csvs,
        params:
            config_path = lambda wildcards: str(workflow.configfiles[-1]),
        shell:
            """
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

            srun -p GPU-H200 \
                --job-name=eval_transcribe \
                --account=efl \
                --gres=gpu:1 \
                --cpus-per-task=4 \
                --mem=32G \
                --time=8:00:00 \
                python {input.script} \
                    --config {params.config_path}
            """

    rule eval_compute_wer:
        input:
            #script = "evaluation/compute_wer.py",
            csvs   = _all_csvs
        output:
            summary = f"{_output_dir}/scores/results_summary.csv",
            latex   = f"{_output_dir}/scores/results.tex",
        params:
            script = "evaluation/compute_wer.py",
            transcriptions_dir = f"{_output_dir}/transcriptions",
            scores_dir         = f"{_output_dir}/scores",
            group_col_arg      = f"--group_col {_eval_cfg['group_col']}" if "group_col" in _eval_cfg else "",
        shell:
            """
            python {params.script} \
                --transcriptions_dir {params.transcriptions_dir} \
                --output_dir         {params.scores_dir} \
                {params.group_col_arg}
            """



# --------------------------------------#
# n-gram LM Evaluation                  #
# --------------------------------------#

if "evaluation" in config and "ngram_lm" in config["evaluation"]:
    _EXP_NAME = config["experiment"]["name"]
    _EXP_DIR  = f"experiments/{_EXP_NAME}"

    _eval_cfg = config["evaluation"]
    _ngram_cfg = _eval_cfg["ngram_lm"]
    _output_dir_ngram = f"{_EXP_DIR}/results-ngram-lm"

    _all_csvs_ngram = [
        f"{_output_dir_ngram}/transcriptions/{d['name']}/{m.get('label', m.get('model', m.get('name', ''))).replace('/', '_')}.csv"
        for m in _eval_cfg["models"]
        for d in _eval_cfg["datasets"]
    ]

    rule eval_transcribe_ngram_lm:
        input:
            script = "evaluation/transcribe_ngram_lm.py",
            # model_ready = f"{_EXP_DIR}/{config['asr_finetuning']['training']['output_dir']}/checkpoint_final.pt"
        output:
            csvs = _all_csvs_ngram
        params:
            config_path = lambda wildcards: str(workflow.configfiles[-1]),
            lm_path = _ngram_cfg["lm_path"],
            alpha = _ngram_cfg.get("alpha", 0.5),
            beta = _ngram_cfg.get("beta", 1.0),
            beam_width = _ngram_cfg.get("beam_width", 100)
        shell:
            """
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

            srun -p GPU-hH200 \
                --job-name=eval_ngram_lm \
                --account=efl \
                --gres=gpu:1 \
                --cpus-per-task=4 \
                --mem=32G \
                --time=8:00:00 \
                python {input.script} \
                    --config {params.config_path} \
                    --use_ngram_lm \
                    --ngram_lm_path {params.lm_path} \
                    --ngram_alpha {params.alpha} \
                    --ngram_beta {params.beta} \
                    --ngram_beam_width {params.beam_width}
            """



    rule eval_compute_wer_ngram_lm:
        input:
            script = "evaluation/compute_wer.py",
            csvs   = _all_csvs_ngram
        output:
            summary = f"{_output_dir_ngram}/scores/results_summary.csv",
            latex   = f"{_output_dir_ngram}/scores/results.tex",
        params:
            transcriptions_dir = f"{_output_dir_ngram}/transcriptions",
            scores_dir         = f"{_output_dir_ngram}/scores",
            group_col_arg      = f"--group_col {_eval_cfg['group_col']}" if "group_col" in _eval_cfg else "",
        shell:
            """
            python {input.script} \
                --transcriptions_dir {params.transcriptions_dir} \
                --output_dir         {params.scores_dir} \
                {params.group_col_arg}
            """