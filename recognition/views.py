from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import List, Dict

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt

import numpy as np
from PIL import Image
import cv2
import face_recognition

from fr.paths import KNN_PATH
from fr.recognize import load_knn
from fr.utils import draw_box_with_label, distance_to_confidence


_KNN_CACHE = {
    'model': None,
    'path': None,
}


def _get_knn(path: Path | None = None):
    global _KNN_CACHE
    knn_path = Path(os.environ.get('FR_KNN_PATH', str(path or KNN_PATH)))
    if _KNN_CACHE['model'] is None or _KNN_CACHE['path'] != knn_path:
        _KNN_CACHE['model'] = load_knn(knn_path)
        _KNN_CACHE['path'] = knn_path
    return _KNN_CACHE['model']


def _process_image(pil_img: Image.Image, detector_model: str = 'hog', upsample: int = 1, threshold: float = 0.6) -> Dict:
    knn = _get_knn()
    rgb = np.array(pil_img.convert('RGB'))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=int(upsample), model=detector_model)
    results: List[Dict] = []
    if boxes:
        encs = face_recognition.face_encodings(rgb, boxes, num_jitters=1)
        preds = knn.predict(encs)
        dists, _ = knn.kneighbors(encs, n_neighbors=1, return_distance=True)
        dists = dists.reshape(-1)
        for box, pred_label, dist in zip(boxes, preds, dists):
            conf = distance_to_confidence(float(dist), float(threshold))
            label = pred_label if dist <= threshold else 'Unknown'
            results.append({'box': box, 'label': str(label), 'distance': float(dist), 'confidence': float(conf)})
            draw_box_with_label(bgr, box, f"{label} {conf:.2f}")

    out_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(out_rgb).save(buf, format='JPEG')
    data_url = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
    return {'image_data_url': data_url, 'results': results}


def index(request: HttpRequest) -> HttpResponse:
    context: Dict = {}
    if request.method == 'POST' and request.FILES.get('image'):
        detector_model = request.POST.get('model', 'hog')
        upsample = int(request.POST.get('upsample', '1'))
        threshold = float(request.POST.get('threshold', '0.6'))
        try:
            img_file = request.FILES['image']
            pil_img = Image.open(img_file)
            out = _process_image(pil_img, detector_model=detector_model, upsample=upsample, threshold=threshold)
            context.update(out)
            context['message'] = f"Detected {len(out['results'])} face(s)."
        except Exception as e:
            context['error'] = str(e)
    return render(request, 'recognition/index.html', context)


@csrf_exempt
def api_recognize(request: HttpRequest) -> JsonResponse:
    if request.method != 'POST' or 'image' not in request.FILES:
        return JsonResponse({'error': 'POST an image file multipart/form-data'}, status=400)
    model = request.POST.get('model', 'hog')
    upsample = int(request.POST.get('upsample', '1'))
    threshold = float(request.POST.get('threshold', '0.6'))
    try:
        pil_img = Image.open(request.FILES['image'])
        out = _process_image(pil_img, detector_model=model, upsample=upsample, threshold=threshold)
        return JsonResponse({'results': out['results']})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def reload_model(request: HttpRequest) -> HttpResponse:
    _KNN_CACHE['model'] = None
    _KNN_CACHE['path'] = None
    return redirect('index')

