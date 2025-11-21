# Makefile for SMORe-Go

# Build output directory
BINDIR := bin

# Go build flags
GOFLAGS := -ldflags="-s -w"

.PHONY: all clean deepwalk node2vec fastrp transe rotate complex sne metapath2vec han ctdne jodie line bpr hpe textgcn skewopt test

all: deepwalk node2vec fastrp transe rotate complex sne metapath2vec han ctdne jodie line bpr hpe textgcn skewopt

# Create bin directory
$(BINDIR):
	mkdir -p $(BINDIR)

# Build DeepWalk
deepwalk: $(BINDIR)
	@echo "Building deepwalk..."
	go build $(GOFLAGS) -o $(BINDIR)/deepwalk ./cmd/deepwalk

# Build Node2Vec
node2vec: $(BINDIR)
	@echo "Building node2vec..."
	go build $(GOFLAGS) -o $(BINDIR)/node2vec ./cmd/node2vec

# Build FastRP
fastrp: $(BINDIR)
	@echo "Building fastrp..."
	go build $(GOFLAGS) -o $(BINDIR)/fastrp ./cmd/fastrp

# Build TransE
transe: $(BINDIR)
	@echo "Building transe..."
	go build $(GOFLAGS) -o $(BINDIR)/transe ./cmd/transe

# Build RotatE
rotate: $(BINDIR)
	@echo "Building rotate..."
	go build $(GOFLAGS) -o $(BINDIR)/rotate ./cmd/rotate

# Build ComplEx
complex: $(BINDIR)
	@echo "Building complex..."
	go build $(GOFLAGS) -o $(BINDIR)/complex ./cmd/complex

# Build SNE
sne: $(BINDIR)
	@echo "Building sne..."
	go build $(GOFLAGS) -o $(BINDIR)/sne ./cmd/sne

# Build Metapath2Vec
metapath2vec: $(BINDIR)
	@echo "Building metapath2vec..."
	go build $(GOFLAGS) -o $(BINDIR)/metapath2vec ./cmd/metapath2vec

# Build HAN
han: $(BINDIR)
	@echo "Building han..."
	go build $(GOFLAGS) -o $(BINDIR)/han ./cmd/han

# Build CTDNE
ctdne: $(BINDIR)
	@echo "Building ctdne..."
	go build $(GOFLAGS) -o $(BINDIR)/ctdne ./cmd/ctdne

# Build JODIE
jodie: $(BINDIR)
	@echo "Building jodie..."
	go build $(GOFLAGS) -o $(BINDIR)/jodie ./cmd/jodie

# Build LINE
line: $(BINDIR)
	@echo "Building line..."
	go build $(GOFLAGS) -o $(BINDIR)/line ./cmd/line

# Build BPR
bpr: $(BINDIR)
	@echo "Building bpr..."
	go build $(GOFLAGS) -o $(BINDIR)/bpr ./cmd/bpr

# Build HPE
hpe: $(BINDIR)
	@echo "Building hpe..."
	go build $(GOFLAGS) -o $(BINDIR)/hpe ./cmd/hpe

# Build TextGCN
textgcn: $(BINDIR)
	@echo "Building textgcn..."
	go build $(GOFLAGS) -o $(BINDIR)/textgcn ./cmd/textgcn

# Build SkewOpt
skewopt: $(BINDIR)
	@echo "Building skewopt..."
	go build $(GOFLAGS) -o $(BINDIR)/skewopt ./cmd/skewopt

# Run tests
test:
	go test -v ./...

# Clean build artifacts
clean:
	rm -rf $(BINDIR)
	go clean

# Install binaries to GOPATH/bin
install:
	go install ./cmd/deepwalk
	go install ./cmd/node2vec
	go install ./cmd/fastrp
	go install ./cmd/transe
	go install ./cmd/rotate
	go install ./cmd/complex
	go install ./cmd/sne
	go install ./cmd/metapath2vec
	go install ./cmd/han
	go install ./cmd/ctdne
	go install ./cmd/jodie
	go install ./cmd/line
	go install ./cmd/bpr
	go install ./cmd/hpe
	go install ./cmd/textgcn
	go install ./cmd/skewopt

# Format code
fmt:
	go fmt ./...

# Run linter
lint:
	golangci-lint run ./...
