package example;

import java.io.IOException;
import java.util.List;

import org.neo4j.procedure.Description;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.UserFunction;

import com.spotify.annoy.ANNIndex;
import com.spotify.annoy.IndexType;

/**
 * This is an example how you can create a simple user-defined function for
 * Neo4j.
 */
public class Annoy {
	
	private static float norm(final float[] u) {
		float n = 0;
		for (float x : u)
			n += x * x;
		return (float) Math.sqrt(n);
	}

	private static float euclideanDistance(final float[] u, final float[] v) {
		float[] diff = new float[u.length];
		for (int i = 0; i < u.length; i++)
			diff[i] = u[i] - v[i];
		return norm(diff);
	}

	@UserFunction
	@Description("example.annoy(filename, dimension, indextype (angular or euclidean), numNeighbors, query item id)")
	public List<Integer> annoy(
			@Name("filename") String indexPath, 
			@Name("dimension") String dim,
			@Name("indextype") String indextype, 
			@Name("numNeighbors") String numNei, 
			@Name("itemId") String item) throws IOException {

		IndexType indexType = null; // 2
		if (indextype.toLowerCase().equals("angular"))
			indexType = IndexType.ANGULAR;
		else if (indextype.toLowerCase().equals("euclidean"))
			indexType = IndexType.EUCLIDEAN;
		else
			throw new RuntimeException("wrong index type specified");
		
		int dimension = Integer.parseInt(dim);
		int queryItem = Integer.parseInt(item);
		
//		long startTime = System.currentTimeMillis();
		ANNIndex annIndex = new ANNIndex(dimension, indexPath, indexType);
//		long constructorTime = System.currentTimeMillis();
		
		// input vector
		float[] u = annIndex.getItemVector(queryItem);
/*		long getItemVectorTime = System.currentTimeMillis();
//		System.out.printf("vector[%d]: ", queryItem);
//
//		for (float x : u) {
//			System.out.printf("%2.2f ", x);
//		}
//		System.out.printf("\n");
*/
		
		int numNeighbors = Integer.parseInt(numNei);
		List<Integer> nearestNeighbors = annIndex.getNearest(u, numNeighbors);
/*		long endTime = System.currentTimeMillis();


		for (int nn : nearestNeighbors) {
			float[] v = annIndex.getItemVector(nn);
			System.out.printf("%d %d %f\n", queryItem, nn,
					(indexType == IndexType.ANGULAR) ? ANNIndex.cosineMargin(u, v) : euclideanDistance(u, v));
		}

		nearestNeighbors.add(Math.toIntExact(constructorTime - startTime));
		nearestNeighbors.add(Math.toIntExact(getItemVectorTime - constructorTime));
		nearestNeighbors.add(Math.toIntExact(endTime - getItemVectorTime));
*/
		annIndex.close();

		return nearestNeighbors;
	}
}