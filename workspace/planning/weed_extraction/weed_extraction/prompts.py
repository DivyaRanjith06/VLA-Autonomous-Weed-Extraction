"""The language prompts a scene can answer, and which weeds each one asks for.

A prompt is valid in a scene only when what it names is there and cannot be read two
ways: "the shepherd weed" needs exactly one shepherd, and "weed 3" needs weeds that
cannot be counted in two orders. Weeds are counted along the rail from the base's home
position. Left and right are the robot's, facing along the rail: left is +y.

The typed prompts are the exception. They name a weed type, which several weeds may
share, and mean the nearest of them.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Prompt:
    kind: str
    text: str
    targets: tuple[str, ...]  # the weeds it names
    every: bool  # True: all the targets must go; False: any one of them will do

    START = -1.7  # the base's home position along the rail, from the home keyframe
    TIE = 0.05  # two weeds closer than this along the rail have no order between them
    # What every sampled scene must be able to answer. The rest -- one weed of a species,
    # the only weed on one side -- depend on the draw, and single-species scenes are
    # wanted too.
    REQUIRED = ("all_weeds", "species_all", "instance", "nearest", "left_bed", "right_bed",
                "crop_safe")
    # The ways to say each typed prompt.
    WORDINGS = {
        "type": ("remove the {species} weed", "pull out the {species} weed",
                 "take the {species} weed out of the bed", "get rid of the {species} weed"),
        "type_side": ("remove the {species} weed from the {side} bed",
                      "pull out the {species} weed in the {side} bed",
                      "take the {species} weed out of the {side} bed",
                      "get rid of the {species} weed in the {side} bed"),
    }

    @classmethod
    def typed(cls, manifest):
        """The prompts that name a weed type, alone or with a bed, in every wording.

        Several weeds may match one. Its target is the match nearest the base's home
        position, the first the base reaches, so the same scene and prompt always lead
        to the same weed.
        """
        found = []
        weeds = sorted(manifest.weeds, key=lambda p: abs(p.x - cls.START))
        for species in sorted({p.species for p in weeds}):
            of = [p for p in weeds if p.species == species]
            places = {None: of, "left": [p for p in of if p.y > 0],
                      "right": [p for p in of if p.y < 0]}
            for side, there in places.items():
                kind = "type_side" if side else "type"
                found += [cls(kind, text.format(species=species, side=side),
                              (there[0].name,), False)
                          for text in cls.WORDINGS[kind] if there]
        return tuple(found)

    @classmethod
    def valid(cls, manifest):
        """Every prompt `manifest` can answer, with its targets."""
        found = []

        def add(kind, text, targets, every=False):
            found.append(cls(kind, text, tuple(p.name for p in targets), every))

        weeds = sorted(manifest.weeds, key=lambda p: p.x)
        beds = {"left": [p for p in weeds if p.y > 0], "right": [p for p in weeds if p.y < 0]}
        middle = sum(bed.x for bed in manifest.beds) / len(manifest.beds)
        if weeds:
            add("all_weeds", "remove all weeds", weeds, every=True)
            add("all_weeds", "remove every weed in the bed", weeds, every=True)
        for species in sorted({p.species for p in weeds}):
            of = [p for p in weeds if p.species == species]
            add("species_all", f"remove all {species} weeds", of, every=True)
            if len(of) == 1:
                add("species_one", f"remove the {species} weed", of)
            for side, there in beds.items():
                if here := [p for p in of if p in there]:
                    add("species_region", f"remove one {species} weed from the {side} bed", here)
                    add("species_area", f"remove all {species} weeds from the {side} bed", here,
                        every=True)
            if front := [p for p in of if p.x < middle]:
                add("species_front", f"remove one {species} weed near the front of the bed",
                    front)
        if weeds and all(b.x - a.x >= cls.TIE for a, b in zip(weeds, weeds[1:])):
            for i, p in enumerate(weeds):
                add("instance", f"remove weed {i + 1}", [p])
            add("nearest", "remove the nearest weed", weeds[:1])
            if manifest.strawberries:
                add("crop_safe", "remove the nearest weed but do not touch the strawberries",
                    weeds[:1])
        for side, there in beds.items():
            if there:
                add(f"{side}_bed", f"remove one weed from the {side} bed", there)
            if len(there) == 1:
                add("side_one", f"remove the weed on the {side} side of the bed", there)
        if weeds and manifest.strawberries:
            add("crop_safe", "remove only the weeds, leave the crop untouched", weeds,
                every=True)
        return tuple(found)

    @classmethod
    def missing(cls, manifest):
        """The REQUIRED kinds `manifest` has no valid prompt for."""
        kinds = {p.kind for p in cls.valid(manifest)}
        return [kind for kind in cls.REQUIRED if kind not in kinds]
