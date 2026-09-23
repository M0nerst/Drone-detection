package com.drone.detector

import android.graphics.RectF

data class Detection(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val score: Float,
    val label: String = "Дрон",
) {
    fun toRectF(): RectF = RectF(left, top, right, bottom)

    val width: Float get() = right - left
    val height: Float get() = bottom - top
    val cx: Float get() = (left + right) / 2f
    val cy: Float get() = (top + bottom) / 2f
}
