# Samples Folder

Place a few example face images here to quickly try the pipeline. Use one folder per person, e.g.:

samples/
  Alice/
    img1.jpg
    img2.jpg
  Bob/
    img1.jpg

Then run the helper to copy them into the dataset structure:

```
python3 scripts/prepare_samples.py
```

With Docker Compose:

```
docker compose run --rm fr python3 scripts/prepare_samples.py
```

After copying:

- Encode: `fr encode`
- Train: `fr train`
- UI: `docker compose up ui` and load `models/knn.joblib`

