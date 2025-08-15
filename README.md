# Hammer and Anvil

Welcome to the code repository for the paper "Hammer and Anvil: A Principled Defense Against Backdoors in Federated Learning".

Below are the instructions for running our experiment code and produce the figures and tables used in our paper.

**Warning**: some of these experiments can take days on state-of-the-art gpus.

This README is divided into the following sections:

- Conda
- UV


## Option 1: Conda (Slow)

### Setup

```bash
conda env create -f environment.yml
``` 

Finally activate the environment:

```bash
conda activate hammer-and-anvil
```

### Run with Conda
Below is the instructions on what to run and what each run allows you to plot/tabulate.

### Table 1
To run the experiments necessary for Table 1, run the commands below:

```bash
./table_1
```

Running this is necessary to create Table 1, Table 3, Figure 4, Figure 6, Figure 7, and Figure 11.


### Table 2
To run the experiments necessary for Table 2, run the commands below:

```bash
./table_2
```

Running this is necessary to create Table 2, Table 3, and Figure 12.

### Code necessary for the remaining figures and tables:

#### Figure 1

```bash
./figure_1
```

#### Figure 2

```bash
./figure_2
```

#### Figure 8

```bash
./figure_8
```

#### Table 5
```bash
./table_5
```

#### Table 6
```bash
./table_6
```

### Other results mentioned in the paper
In the paper, we mention other results that are not necessarily in Tables or Figures. Here is what to run in order for them to be available
when plotting/tabulating with the paper_figures.ipynb notebook.

#### Local training on the fine-tuning set baselines
```bash
./baselines
```

#### Forget-me-not results
```bash
./forget_me_not
```

### Plot/Tabulate
To create the figures and tables, run the notebooks/paper_figures.ipynb. If some of the runs were skipped, make sure not
to run the cells requiring them.


### Option 2: UV (Recommended)

First install UV, one way is through pip:
```bash
pip install uv
```

### Run with UV
Below is the instructions on what to run and what each run allows you to plot/tabulate.

### Table 1
To run the experiments necessary for Table 1, run the commands below:

```bash
uv run ./table_1
```

Running this is necessary to create Table 1, Table 3, Figure 4, Figure 6, Figure 7, and Figure 11.


### Table 2
To run the experiments necessary for Table 2, run the commands below:

```bash
uv run ./table_2
```

Running this is necessary to create Table 2, Table 3, and Figure 12.

### Code necessary for the remaining figures and tables:

#### Figure 1

```bash
uv run ./figure_1
```

#### Figure 2

```bash
uv run ./figure_2
```

#### Figure 8

```bash
uv run ./figure_8
```

#### Table 5
```bash
uv run ./table_5
```

#### Table 6
```bash
uv run ./table_6
```

### Other results mentioned in the paper
In the paper, we mention other results that are not necessarily in Tables or Figures. Here is what to run in order for them to be available
when plotting/tabulating with the paper_figures.ipynb notebook.

#### Local training on the fine-tuning set baselines
```bash
uv run ./baselines
```

#### Forget-me-not results
```bash
uv run ./forget_me_not
```

### Plot/Tabulate
To create the figures and tables, run the notebooks/paper_figures.ipynb. If some of the runs were skipped, make sure not
to run the cells requiring them.
