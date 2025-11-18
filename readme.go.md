# SMORe-Go

A Go implementation of the SMORe (Scalable Modularized Optimization for Recommendation Engines) framework for network embedding and recommendation systems.

## Overview

SMORe-Go is a modern, high-performance Go port of the original SMORe C++ framework. It provides implementations of state-of-the-art graph embedding, knowledge graph embedding, and recommendation algorithms. The framework is designed for scalability and ease of use, with a modular architecture that makes it simple to add new models.

## Requirements

- Go 1.21 or higher
- No external dependencies (pure Go implementation)

## Installation

```bash
git clone https://github.com/cnclabs/smore
cd smore
make -f Makefile.go
```

This will compile all models and place the executables in the `bin/` directory.

## Build Individual Models

You can build individual models using:

```bash
make -f Makefile.go deepwalk    # Build DeepWalk
make -f Makefile.go sasrec      # Build SASRec
make -f Makefile.go textgcn     # Build TextGCN
# ... or any other model
```

## Available Models

### Graph Embedding Models

#### DeepWalk
- **Paper**: [DeepWalk: Online Learning of Social Representations](http://dl.acm.org/citation.cfm?id=2623732) (KDD 2014)
- **Command**: `./bin/deepwalk`
- **Usage**:
```bash
./bin/deepwalk -train net.txt -save embeddings.txt \
  -dimensions 64 -walk_times 10 -walk_steps 40 \
  -window_size 5 -negative_samples 5 -alpha 0.025 -threads 4
```

#### Node2Vec
- **Paper**: [node2vec: Scalable Feature Learning for Networks](https://dl.acm.org/doi/10.1145/2939672.2939754) (KDD 2016)
- **Command**: `./bin/node2vec`
- **Description**: Extends DeepWalk with biased random walks using return parameter p and in-out parameter q

#### LINE
- **Paper**: [LINE: Large-scale Information Network Embedding](http://dl.acm.org/citation.cfm?id=2741093) (WWW 2015)
- **Command**: `./bin/line`
- **Description**: Preserves both first-order and second-order proximity

#### FastRP
- **Paper**: [Fast and Accurate Network Embeddings via Very Sparse Random Projection](https://arxiv.org/abs/1908.11512) (CIKM 2019)
- **Command**: `./bin/fastrp`
- **Description**: Fast random projection-based embedding method

### Heterogeneous Graph Models

#### Metapath2Vec
- **Paper**: [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://dl.acm.org/citation.cfm?id=3098036) (KDD 2017)
- **Command**: `./bin/metapath2vec`
- **Description**: Heterogeneous graph embedding using meta-path-based random walks

#### HAN (Heterogeneous Attention Network)
- **Paper**: [Heterogeneous Graph Attention Network](https://arxiv.org/abs/1903.07293) (WWW 2019)
- **Command**: `./bin/han`
- **Description**: Uses hierarchical attention for heterogeneous graphs

#### TextGCN
- **Paper**: [Graph Convolutional Networks for Text Classification](https://arxiv.org/abs/1809.05679) (AAAI 2019)
- **Command**: `./bin/textgcn`
- **Description**: Graph convolutional network for document-word heterogeneous graphs
- **Usage**:
```bash
./bin/textgcn -train net.txt -field meta.txt -save embeddings.txt \
  -dimensions 64 -sample_times 5 -walk_steps 5 -threads 4
```
- **Field File Format**:
```
doc1 0
doc2 0
word1 2
word2 2
```
Field types: 0=document, 1=filtered, 2=word

### Temporal/Dynamic Graph Models

#### CTDNE
- **Paper**: [Continuous-Time Dynamic Network Embeddings](https://dl.acm.org/citation.cfm?id=3184558.3191526) (WWW 2018)
- **Command**: `./bin/ctdne`
- **Description**: Handles continuous-time dynamic networks

#### JODIE
- **Paper**: [Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks](https://arxiv.org/abs/1908.01207) (KDD 2019)
- **Command**: `./bin/jodie`
- **Description**: Coupled recurrent model for user-item temporal interactions

### Knowledge Graph Embedding Models

#### TransE
- **Paper**: [Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) (NIPS 2013)
- **Command**: `./bin/transe`
- **Description**: Translation-based knowledge graph embedding

#### RotatE
- **Paper**: [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://arxiv.org/abs/1902.10197) (ICLR 2019)
- **Command**: `./bin/rotate`
- **Description**: Rotation-based model in complex vector space

#### ComplEx
- **Paper**: [Complex Embeddings for Simple Link Prediction](http://proceedings.mlr.press/v48/trouillon16.pdf) (ICML 2016)
- **Command**: `./bin/complex`
- **Description**: Complex-valued embeddings for asymmetric relations

### Recommendation Models

#### BPR (Bayesian Personalized Ranking)
- **Paper**: [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://dl.acm.org/citation.cfm?id=1795167) (UAI 2009)
- **Command**: `./bin/bpr`
- **Description**: Matrix factorization with pairwise ranking loss

#### HPE (Heterogeneous Preference Embedding)
- **Paper**: [Query-based Music Recommendations via Preference Embedding](http://dl.acm.org/citation.cfm?id=2959169) (RecSys 2016)
- **Command**: `./bin/hpe`
- **Description**: Handles heterogeneous user-item preferences

#### SASRec (Self-Attentive Sequential Recommendation)
- **Paper**: [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) (ICDM 2018)
- **Command**: `./bin/sasrec`
- **Description**: Transformer-based sequential recommendation
- **Usage**:
```bash
./bin/sasrec -train user_item.txt -save embeddings.txt \
  -dimensions 64 -max_seq_len 50 -num_blocks 2 -num_heads 1 \
  -epochs 10 -batch_size 128 -alpha 0.001 -threads 4
```
- **Input Format**: `user_id item_id` (one interaction per line, chronologically ordered)

#### gSASRec
- **Paper**: [gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling](https://arxiv.org/abs/2308.07192) (RecSys 2023 Best Paper)
- **Command**: `./bin/gsasrec`
- **Description**: Generalized SASRec with Reducing Overconfidence

#### Rec-Denoiser
- **Paper**: [Rec-Denoiser: A Denoising Model for Recommendation](https://dl.acm.org/doi/10.1145/3523227.3546788) (RecSys 2022 Best Paper)
- **Command**: `./bin/recdenoiser`
- **Description**: Adaptively prune noisy items that are unrelated to the next item prediction

#### Skew-Opt
- **Paper**: [Skewness Ranking Optimization for Personalized Recommendation (Skew-OPT)](https://www.researchgate.net/publication/341699675_Skewness_Ranking_Optimization_for_Personalized_Recommendation) (UAI 2020)
- **Command**: `./bin/skewopt`
- **Description**: Skew-OPT framework is a ranking optimization criterion by utilizing the characteristics of skew normal distribution on personalized recommendation task

#### CPR (Cross-Domain Preference Ranking)
- **Paper**: [CPR: Cross-Domain Preference Ranking with User Transformation](https://dl.acm.org/doi/abs/10.1007/978-3-031-28238-6_35) (ECIR 2023)
- **Command**: `./bin/cpr`
- **Description**: Cross-domain recommendation with user transformation, enabling preference prediction across different product categories
- **Usage**:
```bash
./bin/cpr -train_target target.txt -train_source source.txt \
  -save_user users.txt -save_target items_t.txt -save_source items_s.txt \
  -dimensions 64 -update_times 10 -alpha 0.1 -margin 8.0 -threads 4
```
- **Input Format**:
```
# Target domain (e.g., Books)
user1 item1 5.0
user1 item2 4.0

# Source domain (e.g., Movies)
user1 item3 5.0
user1 item4 3.0
```
User IDs must be consistent across both domains for cross-domain learning.

#### TPR (Text-aware Preference Ranking)
- **Paper**: [TPR: Text-aware Preference Ranking for Recommender Systems](http://cfda.csie.org/~cjwang/data/CIKM2020.pdf) (CIKM 2020)
- **Command**: `./bin/tpr`
- **Description**: Combines collaborative filtering with text-based content features for improved recommendations
- **Usage**:
```bash
./bin/tpr -train_ui user_item.txt -train_iw item_word.txt \
  -save_user users.txt -save_item items.txt -save_word words.txt \
  -dimensions 64 -sample_times 10 -text_weight 0.5 -threads 4
```
- **Input Format**:
```
# User-Item interactions
user1 item1
user1 item2
user2 item1

# Item-Word features
item1 word1
item1 word2
item2 word3
```
- **Parameters**:
  - `text_weight`: Balance between collaborative (0.0) and content (1.0) signals (default: 0.5)

### Signed Network Models

#### SNE (Signed Network Embedding)
- **Paper**: [SNE: Signed Network Embedding](https://link.springer.com/chapter/10.1007/978-3-319-57529-2_15) (CIKM 2017)
- **Command**: `./bin/sne`
- **Description**: Handles networks with positive and negative edges

## Input Data Format

### Standard Edge List Format
```
userA itemA 3
userA itemC 5
userB itemA 1
userB itemB 5
userC itemA 4
```

Each line represents an edge: `source_vertex target_vertex [weight]`
- Weight is optional (defaults to 1.0)
- Vertices are identified by strings
- For undirected graphs, you only need to specify each edge once

### Sequential Interaction Format (SASRec, gSASRec)
```
user1 item1
user1 item2
user1 item3
user2 item1
user2 item4
```

Each line represents a user-item interaction in chronological order.

### Field Metadata Format (TextGCN, HAN)
```
vertex_name field_type
```

Field types indicate the vertex type in heterogeneous graphs.

## Output Format

The model saves learned embeddings in the following format:

```
num_vertices dimension
vertex1 0.0815 0.0205 0.2887 0.2965 0.3940
vertex2 -0.2071 -0.2586 0.2332 0.0960 0.2582
vertex3 0.0186 0.1380 0.2136 0.2764 0.4573
...
```

First line: number of vertices and embedding dimension
Following lines: vertex name followed by space-separated embedding values

## Package Structure

```
smore/
├── cmd/                    # Command-line interfaces
│   ├── deepwalk/
│   ├── node2vec/
│   ├── line/
│   ├── bpr/
│   ├── sasrec/
│   ├── textgcn/
│   └── ...
├── internal/models/        # Model implementations
│   ├── deepwalk/
│   ├── node2vec/
│   ├── line/
│   ├── bpr/
│   ├── sasrec/
│   ├── textgcn/
│   └── ...
└── pkg/                    # Reusable packages
    ├── bipartite/         # Bipartite graph utilities
    ├── hetero/            # Heterogeneous graph utilities
    ├── knowledge/         # Knowledge graph utilities
    ├── temporal/          # Temporal graph utilities
    ├── signed/            # Signed network utilities
    ├── pronet/            # Core sampling and optimization
    └── rnn/               # RNN components
```

## Common Command-Line Arguments

Most models support the following common arguments:

- `-train <file>`: Input network/graph file (required)
- `-save <file>`: Output embeddings file (required)
- `-dimensions <int>`: Embedding dimension (default: 64)
- `-threads <int>`: Number of parallel threads (default: 1)
- `-alpha <float>`: Learning rate (default varies by model)
- `-negative_samples <int>`: Number of negative samples (default: 5)
- `-undirected`: Treat edges as undirected (default: true)

Model-specific arguments can be viewed by running the model without arguments:
```bash
./bin/deepwalk
./bin/sasrec
```

## Development

### Running Tests
```bash
make -f Makefile.go test
```

### Code Formatting
```bash
make -f Makefile.go fmt
```

### Installing to GOPATH
```bash
make -f Makefile.go install
```

### Cleaning Build Artifacts
```bash
make -f Makefile.go clean
```

## Performance Tips

1. **Use multiple threads**: Set `-threads` to the number of CPU cores for faster training
2. **Adjust batch size**: For sequential models, larger batch sizes can improve GPU utilization
3. **Tune hyperparameters**: Learning rate and negative samples significantly affect quality
4. **Undirected graphs**: If your graph is undirected, set `-undirected` to avoid duplicate edges

## Citation

If you use SMORe in your research, please cite:

```bibtex
@inproceedings{smore,
  author = {Chen, Chih-Ming and Wang, Ting-Hsiang and Wang, Chuan-Ju and Tsai, Ming-Feng},
  title = {SMORe: Modularize Graph Embedding for Recommendation},
  year = {2019},
  booktitle = {Proceedings of the 13th ACM Conference on Recommender Systems},
  series = {RecSys '19}
}
```

```bibtex
@article{pronet2017,
  title={Vertex-Context Sampling for Weighted Network Embedding},
  author={Chih-Ming Chen and Yi-Hsuan Yang and Yian Chen and Ming-Feng Tsai},
  journal={arXiv preprint arXiv:1711.00227},
  year={2017}
}
```

## Related Work

For more network embedding methods and resources, see [awesome-network-embedding](https://github.com/chihming/awesome-network-embedding).

## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.