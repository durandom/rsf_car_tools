.PHONY: all clean clusters

# Create output directory
out:
	mkdir -p out

# Run clustering analysis
clusters: out
	python ./power_poly.py assets/rsf --select-sample 20 --html out/clusters.html

# Run prediction analysis
predict: out
	python ./power_poly.py assets/rsf --generate --html out/prediction.html

# Default target
all: clusters predict

# Clean generated files
clean:
	rm -rf out/
