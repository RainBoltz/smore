# Makefile for SMORe-Go

# Build output directory
BINDIR := bin

# Go build flags
GOFLAGS := -ldflags="-s -w"

.PHONY: all clean deepwalk line bpr hpe test

all: deepwalk line bpr hpe

# Create bin directory
$(BINDIR):
	mkdir -p $(BINDIR)

# Build DeepWalk
deepwalk: $(BINDIR)
	@echo "Building deepwalk..."
	go build $(GOFLAGS) -o $(BINDIR)/deepwalk ./cmd/deepwalk

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
	go install ./cmd/line
	go install ./cmd/bpr
	go install ./cmd/hpe

# Format code
fmt:
	go fmt ./...

# Run linter
lint:
	golangci-lint run ./...
