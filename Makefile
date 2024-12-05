.PHONY: all clean clusters

# Create output directory
out:
	mkdir -p out

# Run clustering analysis
clusters: out
	python ./power_poly.py assets/rsf --select-sample 20 --html out/clusters.html

# Default target
all: clusters

# Clean generated files
clean:
	rm -rf out/
