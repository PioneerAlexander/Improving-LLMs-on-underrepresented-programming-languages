# Improving LLMs on underrepresented programming languages
**JetBrains test task**

## Experimental setup

Install the requirements for the project as:
```bash
pip install -r requirements.txt
```

### Dataset parser

For the dataset parsing part we firstly cloned the
code from open-source [Kotlin](https://github.com/JetBrains/kotlin) 
project and collected all kotlin filenames from it to [kt_filenames.txt](kt_filenames.txt) using
the following script commands (called in root directory):

```bash
# download Kotlin repo as zip archive
wget https://github.com/JetBrains/kotlin/archive/refs/tags/build-2.0.20-dev-1220.tar.gz

# unzip the archive
tar -xf build-2.0.20-dev-1220.tar.gz

find . -name "*.kt" > kt_filenames.txt
```


The main parser code is implemented in the following [directory](src/parser). The folder contains
the parser script, which can be simply run using the Terminal command

```bash
python3 -m src.parser.parser
```

The body-signature pairs parsed from Kotlin repo will be saved to the "body_signature_pairs.json" file.

In order to create a dataset, use [Kotlin dataset](src/dataset/KotlinCodeCompletionDataset.py) class.
For fixing the train and test part, the special [script](src/dataset/train_test_dataset_split.py) is implemented.
The resulting split files are named "kotlin_test.json" and "kotlin_train.json". In order to get them,
run the command after all previous one.

```bash
python3 -m src.dataset.train_test_dataset_split
```
Utility function for the main models are commonly stored in utils.py files. Functions for processing
the dataset are stored [here](src/dataset/preprocess.py). Extendable test cases for the preprocess functions are provided
in [test](test/dataset) directory.

For the experiments we also need a [CodeXGLUE Code Completion dataset](https://huggingface.co/datasets/microsoft/codexglue_method_generation/tree/main). 
We also created a [custom dataset](src/dataset/CodeXGLUETestDataset.py) for it.

In order to get source json file for creating test dataset, you can simply run:

```bash
wget https://huggingface.co/datasets/microsoft/codexglue_method_generation/resolve/main/test.jsonl
```

### Model finetuning pipeline

The module which is responsible for finetuning is [finetune_model_peft](src/model/finetune_model_peft.py)
It utilizes the HuggingFace PEFT (Parameter Efficient FineTuning). 

One option to run it is executing the command with parameters according to [tyro](https://brentyi.github.io/tyro/) documentation.

```bash
python3 -m src.model.finetune_model_peft phi-1_5.pth phi-1_5-tokenizer/ kotlin_train.json --batch_size=1
```

The first three positional arguments are corresponded
to load model filename, tokenizer load path name and dataset path respectively.
If you do not want to use **Weights&Biases** for logging (which we use by default), add flag `--no-use_wandb`
If you do want, log in by using `wandb login` command.

Note that the loaded model and tokenizer are saved using the special [script](src/model/save_phi-1_5_pretrained.py).
You can run it by:

```bash
python3 -m src.model.save_phi-1_5_pretrained
```

As we finetune only on Kotlin dataset, there is no need to specify the load language mode.

Other run option is using **Weights&Biases** agents. This approach allows to efficiently manage run parameters using
.yaml configs. For better performance some parameters defined in PEFT Config are needed to be finetuned, 
e.g. learning rate, weight decay. Config [here](configs/finetune_phi-1_5.yaml) demonstrates the opportunity to run
experiments with hyperparameter search. It allows also to manage the runs for different models/datasets.

In order to run the script with **Weights&Biases** agent, just run:

```bash
wandb sweep configs/finetune_phi-1_5.yaml --project project_name
wandb agent username/project_name/sweep_id
```
Change project_name to the project name, username to your username you used for logging in to WandB, sweep_id
to id of the created sweep from first command.


### Analysis part

As we run all experiments using **Weights&Biases**, the report which contains all analysis related to this project
is deployed there. You can access it using [this link](https://api.wandb.ai/links/kariakinaleksandr/2wjnpb3c).

In order to run the same evaluation experiments, the [eval_model](src/model/eval_model.py) is implemented.

The experiments for pretrained version of the model we ran using the command

```bash
python3 -m src.model.eval_model phi-1_5.pth phi-1_5-tokenizer/ kotlin kotlin_test.json --batch_size=64 --log_output_every_steps=1
```
for our collected Kotlin dataset, and

```bash
python3 -m src.model.eval_model phi-1_5.pth phi-1_5-tokenizer/ python test.jsonl --batch_size=64 --log_output_every_steps=10
```
for CodeXGLUE Python code completion dataset.

As metrics, we used edit similarity, exact match and BLEU score. They are implemented [here](src/model/metrics.py).
