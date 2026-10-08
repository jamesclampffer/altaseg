# Masker

Attention driven panoptic segmentation for dense ultra high resolution images

## Running

```
MASKER_EMBED_CACHE_DIR=<fast storage>; masker [--host H] [--port P]
```

- `MASKER_EMBED_CACHE_DIR` sets the on-disk embed cache location. Default `~/.masker/embed-cache`.

## System Requirements

Currently optimized for nvidia hardware, with inference falling back to cpu if it has to. This is much, much, slower.

- sam3's running footprint is ~11.6GiB. Maybe possible to run on a 12GB card.
- Multiprompt performance relies on fast virtual memory. NVME with 64GiB of RAM is a good starting point.
  - It'll do prefetch to the gpu to hide latency however more page cache is always better.

## Heirarchical tiling and reprompting

Tile image through SAM3 and reconcile overlaps, redetections, and scaling issues. This makes a handful of assumptions about nature detection of distribution. In practice they seem to be good enough to handle most of the things I've tried.



https://github.com/user-attachments/assets/fdc79af7-6983-4831-a09c-b19adf8aa507



### 1 Split the high resolution into overlapping crops

Crop to native model input resolution, optional scaling. Overlap at 15-20% on all edges.

- High quality detections in the center are assumed likely to be whole
  instance capture and used as exemplars in later recall pass.
- Detections in the overlap bands are tracked but not considered to be a
  candidate instance (or set of them) yet.
- The neck's 72x72x1024 tensors (14x14/image) are cached at this point if they were not sourced from cache initially. The prompt has no bearing on this portion of the computation. The remainder of the forward pass (10-15%) must be recomputed for every prompt.

### 2 Cluster partial detections: overlap band and truncated detections

- Gather closed detections in overlap bands as well as detections that overlap
  from center and truncated by image edge.
- Multiply the bbox of the cluster by some scale factor that has
  demomonstrated good recall; a crop of the scaled bbox will then be run
  through sam again for a second pass.
- Detections fully contained in the overlap band are small. With any luck a
  few partial detections get picked up such that a tight bbox on a single
  instance to give a scale informed pass. Additional new detections will be
  accepted, unioned, upressed, or override a prior detection. The system is
  biased away from the latter.
- Prompt can optionally append exemplar bboxes contained in crop from prior
  pass. Bad if the first pass got a partial or mis-detection.

### 3 Final recall pass based on prior pass detection sizes

- By now it's assumed that at least some detections exist in places there the label should match. Further assume that instances spatially clustered are more likely to be similar sized (+/- ~4-5x). Take crops of prior detections scaled to some multiple of a single detection's bbox. Optionally apply xy translation to get a chance to detect items that would otherwise not be in the instance-centered crop.
  - A better solution to prune redundant crops would help here.
- Further experementation needed to figure out how to pick exemplares from the two prior passes. Detections scores aren't apples to apples.
- The text prompt embeddings may also be passed in here. Sometimes it helps sometimes not. If detections so far have relied on a marginally good enough prompt triggering exemplar redetects this will tend to supress detections.

### Biases and assumptions

#### My use cases:
- masks for photogrammetry based metrology
- semi-supervised training set annotation of crowded environments

#### Design assumptions and requirements:

- Camera with large imager (full/medium frame), good optics, 10-100MP.
  - better images and fewer of them
  - iphone at 48mp raw isn't terrible, either

- Scenes of interest tend to be dense with repeated similar items.
  - warehouses, store shelves, cityscapes, microscopy
  - high instance count per label - 1000-5,000 not uncommon
  - A single high quality capture can be diced into a ton of training images for fine tuning vision models
    - oblique picture of racking -> pallets, items on pallets, bollards, people in situ
    - They'll share common bias - lighting, camera, potential common elements.
    - high resolution + high quality make it possible to crop single or few instance examples.

- Scenes of interest will have a mix of item types. This should capture as much range as possible. Taking pictures is time consuming so what you can out of all of it etc.
  - Efficient reprompt makes this possible. Something like 90% of the fwd
  pass can be cached as 14x14 72x72x1024 bf16 tensors. Reprompts only need
  to hit the later layers.
- Optimized for personal workstation. Some clear paths to parallelize for more powerful hardware.
  - Windows/nvidia, 16gb vram, 96gb ram, low latency nvme
  - minimum sam3 needs ~11.5GB vram to run a pass with batch size 8.


## Scoring against hand-annotated reference set

These are a pain to make.

## Web interface

Workflow:
1) load an image path for start of dir
2) run test propts, tune settings if needed
3) run batch and review + touch-up

Prompt tools:
- text: noun phrase
  - other mechanisms have been secondary thus far
- dragged exemplar bbox
- point (mostly not tested)

Batch mode:
- take current prompt, apply to span of dir

Instance tools:
- refine: re-promopt with right-sized context
- erase: it erases things..
- touch up: brush/erase tools to clean up masks manually

Intra-pass observability:
- cyan - pass 1 - preplanned scaled tiles
- yellow - pass 2 - reprompt over ambiguous clusters, text tokens in prompt only
- magenta - pass 3 - reprompt including exemplar + text tokens

Labels panel:
- This needs work

UI: Little flask server for a portable UI
- keep it simple
- binds to 0.0.0.0, so have fun
- port 5001 by default

## Tests

- `run_tests_fast.ps1` skips forward passes, useful if no gpu
- `run_tests_full.ps1` does everything

## Some Lessions learned

### yak shaving:
- Sometimes it's cool if the project works.
- Still not where I wanted to sink this time.

### compressing bf16 data:
- Split high/low bytes into two contiguous runs. The low bytes have too much entropy to compress well. The upper run contains a lot of zero bits that do compress well. Do that and don't bother with the lower bytes and cat them together for an on-disk format that can be mmapped. Just-in-time decompression is fast and saves bandwidth.

### photogrammetry masking
- Using this to mask dynamic or specular portions of an image makes a marked difference in reconstruction; who would have guessed. This trivializes masking out windows, sky, mobile equipment, people etc. Far fewer images required to get good accuracy in dense point clouds.

