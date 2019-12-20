// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.firebase.samples.apps.mlkit.java.facedetection;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Camera;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import android.util.Log;
import com.google.firebase.samples.apps.mlkit.common.GraphicOverlay;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.automl.FirebaseAutoMLLocalModel;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;
import com.google.firebase.ml.vision.label.FirebaseVisionImageLabel;
import com.google.firebase.ml.vision.label.FirebaseVisionImageLabeler;
import com.google.firebase.ml.vision.label.FirebaseVisionOnDeviceAutoMLImageLabelerOptions;
import com.google.firebase.samples.apps.mlkit.R;
import com.google.firebase.samples.apps.mlkit.common.CameraImageGraphic;
import com.google.firebase.samples.apps.mlkit.common.FrameMetadata;
import com.google.firebase.samples.apps.mlkit.common.GraphicOverlay;
import com.google.firebase.samples.apps.mlkit.common.VisionImageProcessor;
import com.google.firebase.samples.apps.mlkit.java.VisionProcessorBase;
import com.google.firebase.samples.apps.mlkit.java.automl.AutoMLImageLabelerProcessor;
import com.google.firebase.samples.apps.mlkit.java.labeldetector.LabelGraphic;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;

import static java.lang.Math.abs;

/**
 * Face Detector Demo.
 */
public class FaceDetectionProcessor extends VisionProcessorBase<List<FirebaseVisionFace>> {

    private static final String TAG = "FaceDetectionProcessor";

    private final FirebaseVisionFaceDetector detector;

    private final Bitmap overlayBitmap;

    public FaceDetectionProcessor(Resources resources) {
        FirebaseVisionFaceDetectorOptions options =
                new FirebaseVisionFaceDetectorOptions.Builder()
                        .setClassificationMode(FirebaseVisionFaceDetectorOptions.ALL_CLASSIFICATIONS)
                        .setLandmarkMode(FirebaseVisionFaceDetectorOptions.ALL_LANDMARKS)
                        .enableTracking()
                        .build();

        detector = FirebaseVision.getInstance().getVisionFaceDetector(options);

        overlayBitmap = BitmapFactory.decodeResource(resources, R.drawable.clown_nose);
    }

    @Override
    public void stop() {
        try {
            detector.close();
        } catch (IOException e) {
            Log.e(TAG, "Exception thrown while trying to close Face Detector: " + e);
        }
    }

    @Override
    protected Task<List<FirebaseVisionFace>> detectInImage(FirebaseVisionImage image) {
        return detector.detectInImage(image);
    }

    @Override
    protected void onSuccess(
            @Nullable Bitmap originalCameraImage,
            @NonNull List<FirebaseVisionFace> faces,
            @NonNull FrameMetadata frameMetadata,
            @NonNull final GraphicOverlay graphicOverlay) {
        graphicOverlay.clear();
        if (originalCameraImage != null) {
            CameraImageGraphic imageGraphic = new CameraImageGraphic(graphicOverlay, originalCameraImage);
            graphicOverlay.add(imageGraphic);// here for removing the overlay
        }
        for (int i = 0; i < faces.size(); ++i) {
            FirebaseVisionFace face = faces.get(i);
            final int faceId = face.getTrackingId();
            float x = (face.getBoundingBox().centerX());
            float y = face.getBoundingBox().centerY();
            float xOffset = face.getBoundingBox().width() / 2.0f;
            float yOffset = face.getBoundingBox().height() / 2.0f;
            final float left = x - xOffset;
            final float top = y - yOffset;
            final float right = x + xOffset;
            final float bottom = y + yOffset;
            final int j = i;
            if (faceLabelMap.containsKey(faceId))
            {
                String label = faceLabelMap.get(faceId);
                Log.d("kajal",String.valueOf(faceId) + " " + label);
                LabelGraphic labelGraphic = new LabelGraphic(graphicOverlay, label, 0, left, (i%2 == 0) ? top: bottom);
                graphicOverlay.add(labelGraphic);
            }
            else {
                try {
                    Bitmap faceBitmap = Bitmap.createBitmap(originalCameraImage, (int) left, (int) top, (int) (right - left), (int) (bottom - top));
                    float ratio = abs((right - left)/(bottom - top));
                    // scale bitmap a bit
                    faceBitmap = Bitmap.createScaledBitmap(faceBitmap, (int) (50*ratio), 50, true);
                    FirebaseVisionImage img = FirebaseVisionImage.fromBitmap(faceBitmap);
                    detectorLabel.processImage(img)
                            .addOnSuccessListener(new OnSuccessListener<List<FirebaseVisionImageLabel>>() {
                                @Override
                                public void onSuccess(List<FirebaseVisionImageLabel> labels) {
                                    float minConf = 0;
                                    String text = "";
                                    for (FirebaseVisionImageLabel label : labels) {
                                        Log.d("kajal",label.getConfidence() + " : " + label.getText());
                                        if (minConf <= label.getConfidence()) {
                                            minConf = label.getConfidence();
                                            text = label.getText();
                                        }
                                    }
                                    Log.d("kajal","putting: " + String.valueOf(faceId) + " " + text);
                                    faceLabelMap.put(faceId, text);
                                    LabelGraphic labelGraphic = new LabelGraphic(graphicOverlay, text, minConf, left, (j%2 == 0) ? top: bottom);
                                    graphicOverlay.add(labelGraphic);
                                }
                            })
                            .addOnFailureListener(new OnFailureListener() {
                                @Override
                                public void onFailure(@NonNull Exception e) {
                                    Log.d("Kajal", "Label detection failed in autoML");
                                }
                            });
                }
                catch (Exception e)
                {
                    Log.d("kajal", e.toString());
                }
            }
            int cameraFacing =
                    frameMetadata != null ? frameMetadata.getCameraFacing() :
                            Camera.CameraInfo.CAMERA_FACING_BACK;
            FaceGraphic faceGraphic = new FaceGraphic(graphicOverlay, face, cameraFacing, overlayBitmap, originalCameraImage);
            graphicOverlay.add(faceGraphic);
        }
        graphicOverlay.postInvalidate();
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.e(TAG, "Face detection failed " + e);
    }

    private VisionImageProcessor frameProcessor;

    {
        try {
            frameProcessor = new AutoMLImageLabelerProcessor(null, AutoMLImageLabelerProcessor.Mode.LIVE_PREVIEW);
        } catch (FirebaseMLException e) {
            Log.e("Kajal", "Failed to initialize automl" + e);
            e.printStackTrace();
        }
    };

    FirebaseAutoMLLocalModel localModel = new FirebaseAutoMLLocalModel.Builder().setAssetFilePath("automl/manifest.json").build();

    {
        try {
            detectorLabel = FirebaseVision.getInstance().getOnDeviceAutoMLImageLabeler(
                                new FirebaseVisionOnDeviceAutoMLImageLabelerOptions.Builder(localModel)
                                        .setConfidenceThreshold(0)
                                        .build());
        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }
    }

    private FirebaseVisionImageLabeler detectorLabel;
    HashMap<Integer, String> faceLabelMap = new HashMap();
}
