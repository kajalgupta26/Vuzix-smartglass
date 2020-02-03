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
package com.google.firebase.samples.apps.mlkit.java.msapi;

import android.content.res.Resources;
import android.graphics.Bitmap;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import android.os.AsyncTask;
import android.util.Log;
import com.google.firebase.samples.apps.mlkit.common.GraphicOverlay;
import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;
import com.google.firebase.samples.apps.mlkit.R;
import com.google.firebase.samples.apps.mlkit.common.CameraImageGraphic;
import com.google.firebase.samples.apps.mlkit.common.FrameMetadata;
import com.google.firebase.samples.apps.mlkit.java.VisionProcessorBase;
import com.google.firebase.samples.apps.mlkit.java.labeldetector.LabelGraphic;
import com.microsoft.projectoxford.face.FaceServiceRestClient;
import com.microsoft.projectoxford.face.contract.Face;
import com.microsoft.projectoxford.face.contract.FaceRectangle;
import com.microsoft.projectoxford.face.contract.IdentifyResult;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Timer;
import java.util.UUID;
import java.time.Duration;
import java.time.Instant;

/**
 * Face Detector Demo.
 */
public class MSFaceRecognitionProcessor extends VisionProcessorBase<List<FirebaseVisionFace>> {

    private static final String TAG = "FaceDetectionProcessor";

    private final FirebaseVisionFaceDetector detector;

    private static FaceServiceRestClient sFaceClient;

    private String mPersonGroupId;

    private Instant mCurrentTime;
    private Instant mLastTime;
    private boolean isFirstTime;

    public MSFaceRecognitionProcessor(Resources resources, FaceServiceRestClient faceClient) {
        FirebaseVisionFaceDetectorOptions options =
                new FirebaseVisionFaceDetectorOptions.Builder()
                        .setClassificationMode(FirebaseVisionFaceDetectorOptions.ALL_CLASSIFICATIONS)
                        .setLandmarkMode(FirebaseVisionFaceDetectorOptions.ALL_LANDMARKS)
                        .enableTracking()
                        .build();

        detector = FirebaseVision.getInstance().getVisionFaceDetector(options);
        sFaceClient = faceClient;
        mPersonGroupId = resources.getString(R.string.vuzix_group_id);
        faceLabelMap.put("f127d328-72be-4d24-87c3-afe3e811de4a", "kajal");
        faceLabelMap.put("ab96ce12-9394-439c-9d35-8a24d8b78c37", "harsh");
        faceLabelMap.put("6b34fb29-9de4-46b9-87cf-466b24918f74", "Dr. Craig");

        mLastTime = Instant.now();
        isFirstTime = true;
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

        if (faces.size() != 0)
        {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            originalCameraImage.compress(Bitmap.CompressFormat.JPEG, 100, output);
            ByteArrayInputStream inputStream = new ByteArrayInputStream(output.toByteArray());

            mCurrentTime = Instant.now();
            Duration timeElapsed = Duration.between(mLastTime, mCurrentTime);
            if ( timeElapsed.toMillis() >= 6000 || isFirstTime)
            {
                Log.d("kajal", "Detecting new value");
                new DetectionTask(originalCameraImage, graphicOverlay).execute(inputStream);
                mLastTime = mCurrentTime;
                isFirstTime = false;
            }
            else
            {
                Log.d("kajal", "Using old value");
                useOldFaceInfo(graphicOverlay, originalCameraImage);
            }
        }

        graphicOverlay.postInvalidate();
    }

    private void useOldFaceInfo(GraphicOverlay graphicOverlay, Bitmap originalCameraImage)
    {
        graphicOverlay.clear();

        if (originalCameraImage != null) {
            CameraImageGraphic imageGraphic = new CameraImageGraphic(graphicOverlay, originalCameraImage);
            graphicOverlay.add(imageGraphic);// here for removing the overlay
        }

        for (Map.Entry<String, FaceRectangle> entry : faceInfo.entrySet())
        {
            String faceid = entry.getKey();
            FaceRectangle rect = entry.getValue();
            String name = faceName.get(faceid);
            Log.d("kajal", String.valueOf(rect.height) + name);
            if (rect != null && name != null)
            {
                LabelGraphic labelGraphic = new LabelGraphic(graphicOverlay, name, (float) 0.0, (float)rect.left, (float)rect.top);
                graphicOverlay.add(labelGraphic);
            }
        }
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.e("kajal", "Face detection failed " + e);
    }



    // Background task of face detection.
    private class DetectionTask extends AsyncTask<InputStream, String, Face[]> {
        GraphicOverlay graphicOverlay;
        Bitmap originalCameraImage;
        private boolean mSucceed = true;

        DetectionTask(Bitmap _originalCameraImage, GraphicOverlay _graphicOverlay) {
            originalCameraImage = _originalCameraImage;
            graphicOverlay = _graphicOverlay;
        }

        @Override
        protected Face[] doInBackground(InputStream... params)
        {
            // Get an instance of face service client to detect faces in image.
            try{
                // Start detection.
                return sFaceClient.detect(
                        params[0],  /* Input stream of image to detect */
                        true,       /* Whether to return face ID */
                        false,       /* Whether to return face landmarks */
                        /* Which face attributes to analyze, currently we support:
                           age,gender,headPose,smile,facialHair */
                        null);
            }  catch (Exception e) {
                Log.d("Kajal:  MS Face Detection API. failed bruhhh",e.toString());
                useOldFaceInfo(graphicOverlay, originalCameraImage);
                mSucceed = false;
                return null;
            }
        }

        @Override
        protected void onPostExecute(Face[] result) {
            // Show the result on screen when detection is done.
            if (mSucceed)
                afterFaceDetection(result, graphicOverlay, originalCameraImage);
        }
    }

    private void afterFaceDetection(Face[] result, GraphicOverlay graphicOverlay, Bitmap originalCameraImage)
    {
        List<UUID> faceIds = new ArrayList<>();
        faceInfo = new HashMap<>();
        for (Face face:  result) {
            faceIds.add(face.faceId);
            faceInfo.put(face.faceId.toString(), face.faceRectangle);
        }
        Log.d("kajal", String.valueOf(faceInfo.size()));
        new IdentificationTask(mPersonGroupId, graphicOverlay, originalCameraImage).execute(
                faceIds.toArray(new UUID[faceIds.size()]));
    }

    private class IdentificationTask extends AsyncTask<UUID, String, IdentifyResult[]> {
        private boolean mSucceed = true;
        String mPersonGroupId;
        GraphicOverlay graphicOverlay;
        Bitmap originalCameraImage;

        IdentificationTask(String personGroupId, GraphicOverlay _graphicOverlay, Bitmap _originalCameraImage) {
            this.mPersonGroupId = personGroupId;
            graphicOverlay = _graphicOverlay;
            originalCameraImage = _originalCameraImage;
        }

        @Override
        protected IdentifyResult[] doInBackground(UUID... params) {
            // Get an instance of face service client to detect faces in image.
//            try{
//                TrainingStatus trainingStatus = sFaceClient.getLargePersonGroupTrainingStatus(
//                        this.mPersonGroupId);     /* personGroupId */
//                if (trainingStatus.status != TrainingStatus.Status.Succeeded) {
//                    publishProgress("Person group training status is " + trainingStatus.status);
//                    mSucceed = false;
//                    return null;
//                }

            // Start identification.

            try{
                // Start detection.
                return sFaceClient.identityInLargePersonGroup(
                        this.mPersonGroupId,   /* personGroupId */
                        params,                  /* faceIds */
                        1);  /* maxNumOfCandidatesReturned */
            }  catch (Exception e) {
                Log.d("Kajal:  MS Face identification API. failed bruhhh",e.toString());
                useOldFaceInfo(graphicOverlay, originalCameraImage);
                mSucceed = false;
                return null;
            }
        }

        @Override
        protected void onPostExecute(IdentifyResult[] result) {
            // Show the result on screen when detection is done.
            if (mSucceed)
                setUiAfterIdentification(result, graphicOverlay, originalCameraImage);
        }
    }

    private void setUiAfterIdentification(IdentifyResult[] results, GraphicOverlay graphicOverlay, Bitmap originalCameraImage)
    {
        graphicOverlay.clear();

        if (originalCameraImage != null) {
            CameraImageGraphic imageGraphic = new CameraImageGraphic(graphicOverlay, originalCameraImage);
            graphicOverlay.add(imageGraphic);// here for removing the overlay
        }

        for (IdentifyResult result: results)
        {
            String uuid = "unknown";
            if (result.candidates.size() > 0)
                uuid = result.candidates.get(0).personId.toString();
            FaceRectangle rect = faceInfo.get(result.faceId.toString());
            String name = uuid.equals("unknown") ? "unknown" :faceLabelMap.get(uuid) ;
            faceName.put(result.faceId.toString(), name);
            if (rect != null && name != null)
            {
                LabelGraphic labelGraphic = new LabelGraphic(graphicOverlay, name, (float) 0.0, (float)rect.left, (float)rect.top);
                graphicOverlay.add(labelGraphic);
            }

        }
        graphicOverlay.postInvalidate();
    }

    HashMap<String, String> faceLabelMap = new HashMap();
    HashMap<String, FaceRectangle> faceInfo = new HashMap<>();
    HashMap<String, String> faceName = new HashMap<>();
}
