package com.example.adrenotoolstest2

import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.TextView
import com.google.androidgamesdk.GameActivity
import kotlin.math.roundToInt

class MainActivity : GameActivity() {
    companion object {
        init {
            System.loadLibrary("adrenotoolstest2")
        }
    }

    private external fun nativeStats(): String

    private lateinit var statsView: TextView
    private val statsHandler = Handler(Looper.getMainLooper())
    private val statsPoll = object : Runnable {
        override fun run() {
            statsView.text = nativeStats()
            statsHandler.postDelayed(this, 500)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val density = resources.displayMetrics.density
        val padding = (10 * density).roundToInt()
        val background = GradientDrawable().apply {
            setColor(Color.argb(176, 12, 16, 24))
            cornerRadius = 9 * density
        }
        statsView = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 12f
            typeface = Typeface.MONOSPACE
            includeFontPadding = false
            setPadding(padding, padding, padding, padding)
            this.background = background
            isClickable = false
            isFocusable = false
            isFocusableInTouchMode = false
            elevation = 4 * density
        }
        val layout = FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.TOP or Gravity.START,
        ).apply {
            leftMargin = (14 * density).roundToInt()
            topMargin = (14 * density).roundToInt()
        }
        addContentView(statsView, layout)
    }

    override fun onResume() {
        super.onResume()
        statsHandler.removeCallbacks(statsPoll)
        statsPoll.run()
    }

    override fun onPause() {
        statsHandler.removeCallbacks(statsPoll)
        super.onPause()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            hideSystemUi()
        }
    }

    private fun hideSystemUi() {
        val decorView = window.decorView
        decorView.systemUiVisibility = (View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                or View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_FULLSCREEN)
    }
}
