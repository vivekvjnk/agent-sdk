# Google Cloud Storage utils

## list bucket
```bash
gsutil ls -lh gs://<bucket_name>/<dir>
```
eg:
```bash
gsutil ls -lh gs://vhl/schematic_subcircuits
```

## rename files under a augmented dir
```bash example
gsutil -m mv gs://vhl/schematic_subcircuits/* gs://vhl/schematic_subcircuits/images/
```

## upload blobs to certain augmented dir
```bash example
gsutil -m cp -r spice_subcircuits/* gs://vhl/schematic_subcircuits/SPICE/
```
