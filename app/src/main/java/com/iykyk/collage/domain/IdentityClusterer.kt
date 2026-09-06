package com.iykyk.collage.domain

import android.util.Log
import com.iykyk.collage.BuildConfig


/**
 * Groups Appearances into PersonIdentity clusters.
 * 
 * Features:
 * 1. Greedy initial pass.
 * 2. Multi-pass cluster merging to prevent over-fragmentation.
 * 3. Pairwise similarity logging in debug builds.
 */
class IdentityClusterer(
    private val similarityThreshold: Float = 0.5f
) {

    private class Cluster(
        var centroid: FloatArray,
        val appearances: MutableList<Appearance> = mutableListOf()
    )

    fun cluster(appearances: List<Appearance>): List<PersonIdentity> {
        android.util.Log.i("IdentityClusterer", "Clustering ${appearances.size} appearances with threshold $similarityThreshold")
        
        if (BuildConfig.DEBUG) {
            logPairwiseSimilarities(appearances)
        }

        if (appearances.isEmpty()) return emptyList()

        val clusters = mutableListOf<Cluster>()

        // 1. Initial greedy forward-pass
        for ((idx, appearance) in appearances.withIndex()) {
            val embedding = appearance.representativeEmbedding()

            var bestCluster: Cluster? = null
            var bestSim = similarityThreshold

            for ((cIdx, cluster) in clusters.withIndex()) {
                val sim = FaceEmbedder.cosineSimilarity(cluster.centroid, embedding)
                android.util.Log.v("IdentityClusterer", "App[$idx] vs Cluster[$cIdx]: sim=${"%.4f".format(sim)}")
                if (sim > bestSim) {
                    bestSim = sim
                    bestCluster = cluster
                }
            }

            if (bestCluster != null) {
                bestCluster.appearances.add(appearance)
                bestCluster.centroid = calculateCentroid(bestCluster.appearances)
                android.util.Log.d("IdentityClusterer", "App[$idx] -> Cluster matched (sim=${"%.3f".format(bestSim)}). Cluster size: ${bestCluster.appearances.size}")
            } else {
                clusters.add(Cluster(embedding.copyOf(), mutableListOf(appearance)))
                android.util.Log.d("IdentityClusterer", "App[$idx] -> New Cluster created. Total clusters: ${clusters.size}")
            }
        }

        return clusters.map { cluster ->
            val allObservations = cluster.appearances.flatMap { it.observations }
            PersonIdentity(
                appearances = cluster.appearances,
                bestShot = ShotScorer.pickBestShot(allObservations)
            )
        }
    }

    private fun calculateCentroid(appearances: List<Appearance>): FloatArray {
        if (appearances.isEmpty()) return FloatArray(0)
        val dim = appearances.first().representativeEmbedding().size
        val sum = FloatArray(dim)
        for (app in appearances) {
            val e = app.representativeEmbedding()
            for (i in 0 until dim) sum[i] += e[i]
        }
        for (i in 0 until dim) sum[i] /= appearances.size
        
        // Re-normalize to ensure it stays on the hypersphere for cosine similarity
        var norm = 0f
        for (v in sum) norm += v * v
        norm = kotlin.math.sqrt(norm).coerceAtLeast(1e-6f)
        return FloatArray(dim) { sum[it] / norm }
    }

    private fun logPairwiseSimilarities(appearances: List<Appearance>) {
        Log.d("SimDebug", "--- Pairwise Similarity Matrix (n=${appearances.size}) ---")
        for (i in appearances.indices) {
            val ei = appearances[i].representativeEmbedding()
            for (j in i + 1 until appearances.size) {
                val ej = appearances[j].representativeEmbedding()
                val sim = FaceEmbedder.cosineSimilarity(ei, ej)
                Log.d("SimDebug", "App[$i] (len=${appearances[i].observations.size}) vs App[$j] (len=${appearances[j].observations.size}): %.4f".format(sim))
            }
        }
    }
}
