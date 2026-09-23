package com.drone.detector

class BoxTracker(
    private val appearThreshold: Float = 0.50f,
    private val keepThreshold: Float = 0.32f,
    private val maxMisses: Int = 10,
    private val smooth: Float = 0.6f,
) {
    private var last: Detection? = null
    private var misses = 0

    fun update(raw: List<Detection>): List<Detection> {
        val candidate = raw.maxByOrNull { it.score }
        val previous = last
        if (candidate != null) {
            val matched = previous != null && iou(previous, candidate) > 0.15f
            val strong = candidate.score >= appearThreshold
            val keep = matched && candidate.score >= keepThreshold
            if (strong || keep) {
                last = if (previous != null && matched) blend(previous, candidate) else candidate
                misses = 0
                return listOf(last!!)
            }
        }
        if (previous != null && misses < maxMisses) {
            misses++
            return listOf(previous)
        }
        last = null
        misses = 0
        return emptyList()
    }

    private fun blend(prev: Detection, next: Detection): Detection {
        val a = smooth
        val b = 1f - a
        return Detection(
            left = prev.left * a + next.left * b,
            top = prev.top * a + next.top * b,
            right = prev.right * a + next.right * b,
            bottom = prev.bottom * a + next.bottom * b,
            score = next.score,
            label = next.label,
        )
    }
}
