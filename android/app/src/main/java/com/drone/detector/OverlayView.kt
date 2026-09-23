package com.drone.detector

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF39FF14.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x3339FF14
        style = Paint.Style.FILL
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF0A0F14.toInt()
        textSize = 36f
        typeface = Typeface.DEFAULT_BOLD
    }
    private val textBg = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF39FF14.toInt()
        style = Paint.Style.FILL
    }

    @Volatile
    var detections: List<Detection> = emptyList()
        set(value) {
            field = value
            postInvalidate()
        }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        for (det in detections) {
            val rect = det.toRectF()
            canvas.drawRect(rect, fillPaint)
            canvas.drawRect(rect, boxPaint)
            val label = "${det.label} ${"%.0f".format(det.score * 100)}%"
            val tw = textPaint.measureText(label)
            val pad = 10f
            val top = (rect.top - 48f).coerceAtLeast(0f)
            canvas.drawRect(rect.left, top, rect.left + tw + pad * 2, top + 48f, textBg)
            canvas.drawText(label, rect.left + pad, top + 34f, textPaint)
        }
    }
}
