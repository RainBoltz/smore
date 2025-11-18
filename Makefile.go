# Makefile for SMORe-Go

# Build output directory
BINDIR := bin

# Go build flags
GOFLAGS := -ldflags="-s -w"

.PHONY: all clean deepwalk node2vec fastrp transe rotate sne line bpr hpe test

all: deepwalk node2vec fastrp transe rotate sne line bpr hpe

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

# Build SNE
sne: $(BINDIR)
	@echo "Building sne..."
	go build $(GOFLAGS) -o $(BINDIR)/sne ./cmd/sne

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
	go install ./cmd/sne
	go install ./cmd/line
	go install ./cmd/bpr
	go install ./cmd/hpe

# Format code
fmt:
	go fmt ./...

# Run linter
lint:
	golangci-lint run ./...
