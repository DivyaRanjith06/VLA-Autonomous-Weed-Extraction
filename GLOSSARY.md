# Weed Extraction

A simulated robot that removes the weed a person names from a strawberry bed, without touching the crop.

## Language

**Weed type**:
The species of a weed: sowthistle or shepherd.
_Avoid_: Class, category, kind

**Crop**:
The strawberry plants, which must never be touched.

**Bin**:
The container on the robot's base that a removed weed is dropped into.
_Avoid_: Basket, tray, box

**Overview camera**:
The angled camera that follows the robot from the side and sees the bed around it.
_Avoid_: Top view camera, world camera

**Bed camera**:
The camera high above the bed that follows the robot and looks ahead along the bed, showing the weeds the robot is about to reach.
_Avoid_: Top view camera, front camera

**Wrist camera**:
The camera on the robot's hand, giving a colour image and a depth image.

**Prompt**:
The sentence a person gives the robot, naming a weed type and optionally a place in the bed.
_Avoid_: Command, instruction

**Matching weed**:
A weed that fits the prompt. When several match, removing any one of them satisfies the prompt.
_Avoid_: Target

**Nearest matching weed**:
The matching weed closest to the robot along the bed; the one every recorded example removes.

**Wording**:
One of several sentences that mean the same prompt.
_Avoid_: Paraphrase, template

**Contrast pair**:
Two episodes in the same scene whose prompts differ only in the weed type they name.

**Success**:
A matching weed is in the bin and no crop was touched.

**Seen scene**:
A bed layout the model was trained on.

**Held-out scene**:
A bed layout kept out of training, used only to measure the model.

**Baseline**:
The first trained model, using colour images only, that later experiments are compared against.
