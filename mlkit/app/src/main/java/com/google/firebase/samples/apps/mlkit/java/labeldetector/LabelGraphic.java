package com.google.firebase.samples.apps.mlkit.java.labeldetector;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import com.google.firebase.ml.vision.label.FirebaseVisionImageLabel;
import com.google.firebase.samples.apps.mlkit.common.GraphicOverlay;

import java.util.List;

/** Graphic instance for rendering a label within an associated graphic overlay view. */
public class LabelGraphic extends GraphicOverlay.Graphic {

  private final Paint textPaint;
  private final GraphicOverlay overlay;


  public LabelGraphic(GraphicOverlay overlay, String labels, float conf, float x, float y) {
    super(overlay);
    this.overlay = overlay;
    textPaint = new Paint();
    textPaint.setColor(Color.WHITE);
    textPaint.setTextSize(35.0f);
    minConf = conf;
    text = labels;
    this.x = translateForVuzix(translateX(x), false);
    this.y = translateForVuzix(translateY(y), true);
    postInvalidate();
  }

  @Override
  public synchronized void draw(Canvas canvas) {
    canvas.drawText(text , x, y, textPaint);
  }

  float minConf = 0;
  String text = "";
  float x, y;
}
