# Copyright 2026 Jim Clampffer. Created 2026-10-07.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""annotations, mostly coco internal representation"""

from __future__ import annotations

from masker.io.coco import coco_t


def ids_for_labels(dataset: coco_t, labels: list[str]) -> set[int]:
    """Category ids for the named labels; KeyError on an unknown name."""
    by_name: dict[str, set[int]] = {}
    for cat in dataset["categories"]:
        by_name.setdefault(cat["name"], set()).add(cat["id"])
    unknown = sorted(set(labels) - set(by_name))
    if unknown:
        raise KeyError("labels not in dataset: {}; available: {}".format(unknown, sorted(by_name)))
    return {cid for label in labels for cid in by_name[label]}


def _supercategory_by_name(dataset: coco_t) -> dict[str, str]:
    """name -> supercategory"""
    out: dict[str, str] = {}
    for c in dataset["categories"]:
        out.setdefault(c["name"], c.get("supercategory") or "")
    return out


def drop_labels(dataset: coco_t, labels: list[str]) -> coco_t:
    """Remove labels from categories and every annotation using them."""
    drop_ids = ids_for_labels(dataset, labels)
    dropped = {c["name"] for c in dataset["categories"] if c["id"] in drop_ids}
    parent_of = _supercategory_by_name(dataset)

    def reroot(superc: str) -> str:
        seen: set[str] = set()
        while superc in dropped:
            if superc in seen:
                return ""  # dropped cycle: nothing left to re-root to
            seen.add(superc)
            superc = parent_of.get(superc, "")
        return superc

    categories = []
    for c in dataset["categories"]:
        if c["id"] in drop_ids:
            continue
        if (c.get("supercategory") or "") in dropped:
            c = {**c, "supercategory": reroot(c.get("supercategory") or "")}
        categories.append(c)
    return {
        **dataset,
        "categories": categories,
        "annotations": [a for a in dataset["annotations"] if a["category_id"] not in drop_ids],
    }


def add_labels(dataset: coco_t, labels: list[str]) -> coco_t:
    existing = {c["name"] for c in dataset["categories"]}
    dupes = sorted({name for name in labels if name in existing})
    if dupes:
        raise ValueError("labels already in dataset: {}".format(dupes))
    next_id = max((c["id"] for c in dataset["categories"]), default=0) + 1
    added = [
        {"id": next_id + i, "name": name, "supercategory": ""}
        for i, name in enumerate(labels)
    ]
    return {**dataset, "categories": dataset["categories"] + added}


def merge_labels(dataset: coco_t, labels: list[str], new_name: str) -> coco_t:
    source_ids = ids_for_labels(dataset, labels)
    target = next((c for c in dataset["categories"] if c["name"] == new_name), None)
    if target is None:
        target_id = min(source_ids)
    else:
        target_id = target["id"]
    remap_ids = source_ids - {target_id}
    vanished = {
        c["name"] for c in dataset["categories"]
        if c["id"] in source_ids and c["name"] != new_name
    }
    parent_of = _supercategory_by_name(dataset)

    def survivor_parent(superc: str) -> str:
        # nearest pre-merge ancestor outside the merge set
        seen: set[str] = set()
        while superc in vanished or superc == new_name:
            if superc in seen:
                return ""
            seen.add(superc)
            superc = parent_of.get(superc, "")
        return superc

    categories = []
    for c in dataset["categories"]:
        if c["id"] in remap_ids:
            continue
        superc = c.get("supercategory") or ""
        if c["id"] == target_id:
            c = {**c, "name": new_name}
            if superc in vanished or superc == new_name:
                c["supercategory"] = survivor_parent(superc)
        elif superc in vanished:
            c = {**c, "supercategory": new_name}
        categories.append(c)
    annotations = [
        {**a, "category_id": target_id} if a["category_id"] in remap_ids else a
        for a in dataset["annotations"]
    ]
    return {**dataset, "categories": categories, "annotations": annotations}


def category_tree(dataset: coco_t) -> list[dict]:
    cats = dataset["categories"]
    id_by_name: dict[str, int] = {}
    for c in cats:
        id_by_name.setdefault(c["name"], c["id"])
    parent: dict[int, int | None] = {}  # None for roots and group members
    for c in cats:
        parent[c["id"]] = id_by_name.get(c.get("supercategory") or "")
    for start in parent:  # cut supercategory cycles so every chain terminates
        chain: set[int] = set()
        cur = start
        while parent[cur] is not None:
            if cur in chain:
                parent[cur] = None
                break
            chain.add(cur)
            cur = parent[cur]
    nodes = {c["id"]: {"id": c["id"], "name": c["name"], "children": []} for c in cats}
    roots: list[dict] = []
    groups: dict[str, dict] = {}
    for c in cats:
        node = nodes[c["id"]]
        if parent[c["id"]] is not None:
            nodes[parent[c["id"]]]["children"].append(node)
            continue
        superc = c.get("supercategory") or ""
        if superc and superc not in id_by_name:  # group heading
            group = groups.get(superc)
            if group is None:
                group = groups[superc] = {"id": None, "name": superc, "children": []}
                roots.append(group)
            group["children"].append(node)
        else:
            roots.append(node)
    return roots


def set_category_parent(dataset: coco_t, name: str, parent: str) -> coco_t:
    names = {c["name"] for c in dataset["categories"]}
    if name not in names:
        raise ValueError("no category named {!r}".format(name))
    if parent:
        headings = {c.get("supercategory") or "" for c in dataset["categories"]} - {""}
        if parent not in names and parent not in headings:
            raise ValueError(
                "no category or group heading named {!r} to use as parent".format(parent)
            )
        if parent == name:
            raise ValueError("{!r} cannot be its own parent".format(name))
        parent_of = _supercategory_by_name(dataset)
        seen: set[str] = set()
        cur = parent
        while cur in parent_of and cur not in seen:  # walk parent's ancestors
            seen.add(cur)
            cur = parent_of[cur]
            if cur == name:
                raise ValueError(
                    "{!r} descends from {!r}; re-parenting would create a cycle".format(parent, name)
                )
    return {
        **dataset,
        "categories": [
            {**c, "supercategory": parent} if c["name"] == name else c
            for c in dataset["categories"]
        ],
    }
