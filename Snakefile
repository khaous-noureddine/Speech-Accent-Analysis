"""
Constrastive Learning for Accented Speech Recognition
"""

# --------------------------------------#
# Data Preparation                      #
# --------------------------------------#

if "speech_accents" in config.get("data_preparation", {}):
    rule prepare_speech_accent_corpus:
        input:
            data_dir       = config["data_preparation"]["speech_accents"]["raw_data_dir"]
        output:
            parquet        = config["data_preparation"]["speech_accents"]["parquet_path"]
        params:
            output_dir     = config["data_preparation"]["speech_accents"]["processed_data_dir"],
            script         = workflow.basedir + "/corpus/import_speech_accent.py"
        shell:
            """
            pixi run python {params.script} \
                --corpus_dir {input.data_dir} \
                --output_parquet {output.parquet} \
                --audio_dir {params.output_dir}/wavs
            """


if "arctic" in config.get("data_preparation", {}):
    rule prepare_arctic_corpus:
        input:
            data_dir       = config["data_preparation"]["arctic"]["raw_data_dir"]
        output:
            parquet        = config["data_preparation"]["arctic"]["parquet_path"]
        params:
            output_dir     = config["data_preparation"]["arctic"]["processed_data_dir"],
            script         = workflow.basedir + "/corpus/import_arctic.py"
        shell:
            """
            pixi run python {params.script} \
                --corpus_dir {input.data_dir} \
                --output_parquet {output.parquet} \
                --audio_dir {params.output_dir}/wavs
            """


if "l2_arctic" in config.get("data_preparation", {}):
    rule prepare_l2_arctic_corpus:
        input:
            data_dir       = config["data_preparation"]["l2_arctic"]["raw_data_dir"]
        output:
            parquet        = config["data_preparation"]["l2_arctic"]["parquet_path"]
        params:
            output_dir     = config["data_preparation"]["l2_arctic"]["processed_data_dir"],
            script         = workflow.basedir + "/corpus/import_l2_arctic.py"
        shell:
            """
            pixi run python {params.script} \
                --corpus_dir {input.data_dir} \
                --output_parquet {output.parquet} \
                --audio_dir {params.output_dir}/wavs
            """


if "librispeech" in config.get("data_preparation", {}):
    rule prepare_librispeech_train:
        input:
            data_dir = config["data_preparation"]["librispeech"]["train_raw_dir"]
        output:
            parquet  = config["data_preparation"]["librispeech"]["parquet_train_path"]
        params:
            audio_dir = config["data_preparation"]["librispeech"]["processed_train_data_dir"] + "/wavs",
            script    = workflow.basedir + "/corpus/import_librispeech.py"
        shell:
            """
            pixi run python {params.script} \
                --corpus_dir {input.data_dir} \
                --output_parquet {output.parquet} \
                --audio_dir {params.audio_dir} \
                --split train
            """

    rule prepare_librispeech_eval:
        input:
            data_dir = config["data_preparation"]["librispeech"]["eval_raw_dir"]
        output:
            parquet  = config["data_preparation"]["librispeech"]["parquet_eval_path"]
        params:
            audio_dir = config["data_preparation"]["librispeech"]["processed_eval_data_dir"] + "/wavs",
            script    = workflow.basedir + "/corpus/import_librispeech.py"
        shell:
            """
            pixi run python {params.script} \
                --corpus_dir {input.data_dir} \
                --output_parquet {output.parquet} \
                --audio_dir {params.audio_dir} \
                --split eval
            """

    rule prepare_librispeech_test:
        input:
            data_dir = config["data_preparation"]["librispeech"]["test_raw_dir"]
        output:
            parquet  = config["data_preparation"]["librispeech"]["parquet_test_path"]
        params:
            audio_dir = config["data_preparation"]["librispeech"]["processed_test_data_dir"] + "/wavs",
            script    = workflow.basedir + "/corpus/import_librispeech.py"
        shell:
            """
            pixi run python {params.script} \
                --corpus_dir {input.data_dir} \
                --output_parquet {output.parquet} \
                --audio_dir {params.audio_dir} \
                --split test
            """


if "speech_accents" in config.get("data_preparation", {}):
    rule prepare_speech_accent_corpus:
        input:
            data_dir       = config["data_preparation"]["speech_accents"]["raw_data_dir"]
        output:
            parquet        = config["data_preparation"]["speech_accents"]["parquet_path"]
        params:
            output_dir     = config["data_preparation"]["speech_accents"]["processed_data_dir"],
            script         = workflow.basedir + "/corpus/import_speech_accent.py"
        shell:
            """
            pixi run python {params.script} \
                --corpus_dir {input.data_dir} \
                --output_parquet {output.parquet} \
                --audio_dir {params.output_dir}/wavs
            """


if "edacc_validation" in config.get("data_preparation", {}):
    rule prepare_edacc_validation:
        input:
            cache_dir = config["data_preparation"]["edacc_validation"]["raw_data_dir"]
        output:
            parquet   = config["data_preparation"]["edacc_validation"]["parquet_path"]
        params:
            audio_dir = config["data_preparation"]["edacc_validation"]["processed_data_dir"] + "/wavs",
            split     = config["data_preparation"]["edacc_validation"]["split"],
            script    = workflow.basedir + "/corpus/import_edacc.py"
        shell:
            """
            python {params.script} \
                --cache_dir      {input.cache_dir} \
                --output_parquet {output.parquet} \
                --audio_dir      {params.audio_dir} \
                --split          {params.split}
            """

if "edacc_test" in config.get("data_preparation", {}):
    rule prepare_edacc_test:
        input:
            cache_dir = config["data_preparation"]["edacc_test"]["raw_data_dir"]
        output:
            parquet   = config["data_preparation"]["edacc_test"]["parquet_path"]
        params:
            audio_dir = config["data_preparation"]["edacc_test"]["processed_data_dir"] + "/wavs",
            split     = config["data_preparation"]["edacc_test"]["split"],
            script    = workflow.basedir + "/corpus/import_edacc.py"
        shell:
            """
            python {params.script} \
                --cache_dir      {input.cache_dir} \
                --output_parquet {output.parquet} \
                --audio_dir      {params.audio_dir} \
                --split          {params.split}
            """
    
    
if "aesrc" in config.get("data_preparation", {}):
    _aesrc_cfg = config["data_preparation"]["aesrc"]

    rule prepare_aesrc_corpus:
        input:
            script   = workflow.basedir + "/corpus/import_aesrc.py",
            data_dir = _aesrc_cfg["raw_data_dir"]
        output:
            parquet        = _aesrc_cfg["parquet_path"],
            csv            = _aesrc_cfg.get("csv_path", ""),
            latex_stats    = _aesrc_cfg.get("latex_stats_path", ""),
            overlap_report = _aesrc_cfg.get("overlap_report_path", ""),
            accent_stats   = _aesrc_cfg.get("accent_stats_txt", "")
        params:
            audio_dir  = _aesrc_cfg["processed_data_dir"] + "/wavs",
            dev_ratio  = _aesrc_cfg.get("dev_ratio", 0.10),
            seed       = _aesrc_cfg.get("seed", 42)
        shell:
            r"""
            OUTPUT_CSV_ARG=""
            if [ -n "{output.csv}" ]; then
                OUTPUT_CSV_ARG="--output_csv {output.csv}"
            fi

            OUTPUT_LATEX_ARG=""
            if [ -n "{output.latex_stats}" ]; then
                OUTPUT_LATEX_ARG="--output_latex {output.latex_stats}"
            fi

            OVERLAP_REPORT_ARG=""
            if [ -n "{output.overlap_report}" ]; then
                OVERLAP_REPORT_ARG="--overlap_report {output.overlap_report}"
            fi

            ACCENT_STATS_ARG=""
            if [ -n "{output.accent_stats}" ]; then
                ACCENT_STATS_ARG="--accent_stats_txt {output.accent_stats}"
            fi

            python {input.script} \
                --corpus_dir     {input.data_dir} \
                --output_parquet {output.parquet} \
                --audio_dir      {params.audio_dir} \
                --dev_ratio      {params.dev_ratio} \
                --seed           {params.seed} \
                $OUTPUT_CSV_ARG \
                $OUTPUT_LATEX_ARG \
                $OVERLAP_REPORT_ARG \
                $ACCENT_STATS_ARG
            """

# --------------------------------------#
# Supervised Contrastive Learning       #
# --------------------------------------#

if "supervised_contrastive_training" in config:
    rule supervised_contrastive_training:
        input:
            script            = "stage2/supcon_train.py",
            arctic_parquet    = config["supervised_contrastive_training"]["data"]["arctic_parquet_path"],
            l2_arctic_parquet = config["supervised_contrastive_training"]["data"]["l2_arctic_parquet_path"]
        output:
            checkpoint = f"checkpoints/conditions/A/stage_2/checkpoint_epoch{config['supervised_contrastive_training']['training']['epochs']:03d}.pt"
        params:
            save_dir              = config["supervised_contrastive_training"]["training"]["checkpoint_dir"],
            sample_rate           = config["supervised_contrastive_training"]["data"]["sample_rate"],
            max_audio_len_s       = config["supervised_contrastive_training"]["data"]["max_audio_len_s"],
            number_of_workers     = config["supervised_contrastive_training"]["sampler"]["number_of_workers"],
            k_utterances          = config["supervised_contrastive_training"]["sampler"]["k_utterances"],
            s_speakers            = config["supervised_contrastive_training"]["sampler"]["s_speakers"],
            n_batches             = config["supervised_contrastive_training"]["sampler"]["n_batches"],
            seed                  = config["supervised_contrastive_training"]["sampler"]["seed"],
            model_name            = config["supervised_contrastive_training"]["model"]["model_name"],
            proj_hidden_dim       = config["supervised_contrastive_training"]["model"]["proj_hidden_dim"],
            proj_out_dim          = config["supervised_contrastive_training"]["model"]["proj_out_dim"],
            vocab_size            = config["supervised_contrastive_training"]["model"]["vocab_size"],
            min_frozen_layer      = config["supervised_contrastive_training"]["model"]["min_frozen_layer"],
            max_frozen_layer      = config["supervised_contrastive_training"]["model"]["max_frozen_layer"],
            ctc_lambda            = config["supervised_contrastive_training"]["model"]["ctc_lambda"],
            temperature           = config["supervised_contrastive_training"]["model"]["temperature"],
            epochs                = config["supervised_contrastive_training"]["training"]["epochs"],
            learning_rate         = config["supervised_contrastive_training"]["training"]["learning_rate"],
            warmup_steps          = config["supervised_contrastive_training"]["training"]["warmup_steps"],
            use_ctc               = config["supervised_contrastive_training"]["training"]["use_ctc"],
            tokenizer             = config["supervised_contrastive_training"]["training"]["tokenizer"],
            device                = config["supervised_contrastive_training"]["training"]["device"],
            tensorboard_dir       = config["supervised_contrastive_training"]["training"]["tensorboard_dir"],
            save_every_n_epochs   = config["supervised_contrastive_training"]["training"]["save_every_n_epochs"],
            use_mixed_precision   = config["supervised_contrastive_training"]["training"]["use_mixed_precision"],
            eval_every_n_epochs   = config["supervised_contrastive_training"]["evaluation"]["eval_every_n_epochs"],
            eval_n_neg_samples    = config["supervised_contrastive_training"]["evaluation"]["eval_n_neg_samples"],
            eval_batch_size       = config["supervised_contrastive_training"]["evaluation"]["eval_batch_size"],
            retrieval_ks          = lambda wildcards: " ".join(map(str, config["supervised_contrastive_training"]["evaluation"]["retrieval_ks"])),
            eval_metrics          = lambda wildcards: " ".join(config["supervised_contrastive_training"]["evaluation"]["eval_metrics"]),
        shell:
            """
            export LD_PRELOAD={workflow.basedir}/.pixi/envs/default/lib/libstdc++.so.6
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

            srun -p GPU-H200 \
                --job-name=supcon_train \
                --account=efl \
                --gres=gpu:1 \
                --cpus-per-task=4 \
                --mem=128G \
                --time=3-00:00:00 \
                python {input.script} \
                    --arctic_parquet_path    {input.arctic_parquet} \
                    --l2_arctic_parquet_path {input.l2_arctic_parquet} \
                    --sample_rate            {params.sample_rate} \
                    --max_audio_len_s        {params.max_audio_len_s} \
                    --num_workers            {params.number_of_workers} \
                    --k_utterances           {params.k_utterances} \
                    --s_speakers             {params.s_speakers} \
                    --n_batches              {params.n_batches} \
                    --seed                   {params.seed} \
                    --model_name             {params.model_name} \
                    --proj_hidden_dim        {params.proj_hidden_dim} \
                    --proj_out_dim           {params.proj_out_dim} \
                    --vocab_size             {params.vocab_size} \
                    --min_frozen_layer       {params.min_frozen_layer} \
                    --max_frozen_layer       {params.max_frozen_layer} \
                    --ctc_lambda             {params.ctc_lambda} \
                    --temperature            {params.temperature} \
                    --epochs                 {params.epochs} \
                    --lr                     {params.learning_rate} \
                    --warmup_steps           {params.warmup_steps} \
                    --use_ctc                {params.use_ctc} \
                    --tokenizer              {params.tokenizer} \
                    --device                 {params.device} \
                    --save_dir               {params.save_dir} \
                    --save_every_n_epochs    {params.save_every_n_epochs} \
                    --tensorboard_dir        {params.tensorboard_dir} \
                    --use_mixed_precision    {params.use_mixed_precision} \
                    --eval_every_n_epochs    {params.eval_every_n_epochs} \
                    --eval_n_neg_samples     {params.eval_n_neg_samples} \
                    --eval_batch_size        {params.eval_batch_size} \
                    --retrieval_ks           {params.retrieval_ks} \
                    --eval_metrics           {params.eval_metrics}
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
# Evaluation                            #
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
            model_ready = f"{_EXP_DIR}/{config['asr_finetuning']['training']['output_dir']}/checkpoint_final.pt"
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
            script = "evaluation/compute_wer.py",
            csvs   = _all_csvs
        output:
            summary = f"{_output_dir}/scores/results_summary.csv",
            latex   = f"{_output_dir}/scores/results.tex",
        params:
            transcriptions_dir = f"{_output_dir}/transcriptions",
            scores_dir         = f"{_output_dir}/scores",
            group_col_arg      = f"--group_col {_eval_cfg['group_col']}" if "group_col" in _eval_cfg else "",
        shell:
            """
            python {input.script} \
                --transcriptions_dir {params.transcriptions_dir} \
                --output_dir         {params.scores_dir} \
                {params.group_col_arg}
            """