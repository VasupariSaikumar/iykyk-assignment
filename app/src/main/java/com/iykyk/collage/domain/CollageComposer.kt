package com.iykyk.collage.domain

import android.graphics.*
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

object CollageComposer {

    private const val TILE_SIZE = 480
    private const val PADDING = 16
    private const val CORNER_RADIUS = 28f

    /**
     * Crops generously around the face bbox for display -- per the spec, a tight bbox crop
     * looks like a low-res mugshot. Expanding by ~80% on each side while staying in-bounds
     * gives a "shoulders up" portrait look instead.
     */
    fun generousCrop(source: Bitmap, bbox: Rect): Bitmap {
        val expandX = (bbox.width() * 0.8f).toInt()
        val expandY = (bbox.height() * 0.9f).toInt() // extra headroom above/below
        val left = max(0, bbox.left - expandX)
        val top = max(0, bbox.top - expandY)
        val right = min(source.width, bbox.right + expandX)
        val bottom = min(source.height, bbox.bottom + expandY)

        // Keep it square-ish for a clean grid tile.
        val w = right - left
        val h = bottom - top
        val side = max(w, h)
        val cx = (left + right) / 2
        val cy = (top + bottom) / 2
        val sqLeft = (cx - side / 2).coerceIn(0, source.width - 1)
        val sqTop = (cy - side / 2).coerceIn(0, source.height - 1)
        val sqSide = min(side, min(source.width - sqLeft, source.height - sqTop)).coerceAtLeast(1)

        val cropped = Bitmap.createBitmap(source, sqLeft, sqTop, sqSide, sqSide)
        val scaled = Bitmap.createScaledBitmap(cropped, TILE_SIZE, TILE_SIZE, true)
        if (cropped != source) cropped.recycle() // Clean up intermediate crop
        return scaled
    }

    private fun roundedTile(bitmap: Bitmap): Bitmap {
        val output = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        val rect = RectF(0f, 0f, bitmap.width.toFloat(), bitmap.height.toFloat())
        canvas.drawRoundRect(rect, CORNER_RADIUS, CORNER_RADIUS, paint)
        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_IN)
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        return output
    }

    /**
     * Simple responsive grid (2 or 3 columns depending on count) -- an easy, presentable
     * stand-in for the "Instagram Story collage" reference; swap in a nicer bespoke layout
     * if you have time left after the accuracy-critical pieces are solid.
     */
    fun composeCollage(identities: List<PersonIdentity>): Bitmap {
        val columns = if (identities.size <= 4) 2 else 3
        val rows = ceil(identities.size / columns.toFloat()).toInt()

        val width = columns * TILE_SIZE + (columns + 1) * PADDING
        val height = rows * (TILE_SIZE + 56) + (rows + 1) * PADDING // +56 for the label under each tile

        val output = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)
        canvas.drawColor(Color.parseColor("#111111"))

        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 34f
            textAlign = Paint.Align.CENTER
        }

        identities.forEachIndexed { index, identity ->
            val col = index % columns
            val row = index / columns
            val x = PADDING + col * (TILE_SIZE + PADDING)
            val y = PADDING + row * (TILE_SIZE + 56 + PADDING)

            val rounded = roundedTile(identity.bestShot.faceCrop)
            canvas.drawBitmap(rounded, x.toFloat(), y.toFloat(), null)

            canvas.drawText(
                "${identity.appearanceCount}x appearances",
                x + TILE_SIZE / 2f,
                y + TILE_SIZE + 40f,
                textPaint
            )
        }

        return output
    }
}
