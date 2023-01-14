# ClusterClue
A data pipeline to detect Star Clusters using Gaia DR3. The detection method relies on two main steps that combine unsupervised and supervised learning. First, the clustering algorithm HDBSCAN searches cluster candidates using the astrometric position of stars. Second, a trained neural network validates a candidate exploding its CMD.  

## Files Description
- nn_weights/: weights of the trained CNN that validates clusters.
- src/functions.py: functions that define the pipeline.
- src/examples.py: example of how to use ClusterClue.

## Author
Truman Tapia

## License
MIT license, as found in the LICENSE file. 
