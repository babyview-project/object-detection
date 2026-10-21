# Object boxes and formats in BabyView frames: instructions for Mira and Emma

Thank you for helping with this! You will check and correct object boxes on single frames
from BabyView, the head-camera videos that children wore at home. Each frame already has
some boxes drawn by computer models. Your job is to make every frame correct and complete:
box **every object in view** and give each box a **name** in your own words (there is no
fixed list), fix or delete wrong boxes, add the objects the models missed, and say for every
box whether it is the real object, a toy, a picture or an image on a screen.

Your corrected frames become our new "gold set": the reference we use to measure how well
two detection methods work, and to check a model's format labels. Accuracy and consistency
matter much more than speed.

---

## 1. Privacy rules (read first)

These frames show the inside of families' homes, children and their family members.

- Work only on a lab computer or a computer the lab has approved for BabyView data.
- Get your task folders only from the lab's IRB-approved storage folder that Jane shares
  with you, and put your exports back there. **Never** send frames or exports by email,
  Slack, text message, Google Drive outside the lab folder, GitHub or any other website.
- No screenshots or photos of the screen. If you want to ask about a frame, send Jane the
  **file name** (for example `04529aebb0a3.jpg`) and describe what you see.
- CVAT runs on your own computer (`localhost`); nothing is uploaded to the internet.
- When a batch is finished and exported, delete your local copy of the task folder after
  Jane confirms she has your export.

---

## 2. What you will get

For each batch, a zip file named like `Mira_batch1.zip`. Unzipped, it contains:

| File | What it is |
|---|---|
| `images/` | the frames, 512 × 910 pixels, named with random ids (e.g. `005f24388ca4.jpg`) |
| `labels.json` | the label setup: one label, **object**, with two attributes, **name** and **format** |
| `annotations.zip` | the pre-drawn boxes, to load into the task |

Batch 1 has 200 frames each. Later batches have 100 frames (batch 5 is a bit smaller).
Some frames are given to both of you so we can measure agreement. You won't know which
ones, so please **do not compare or discuss specific frames with each other**; ask Jane
instead.

---

## 3. Setting up CVAT (once per computer)

CVAT is a free annotation program that runs in your web browser, on your own computer.
If CVAT is already installed on the computer you use (Mira: the one from last year's
annotation), skip to 3.3 and log in with your existing account.

### 3.1 Install

1. Install **Docker Desktop** (docker.com) and start it. On Windows, accept the WSL 2
   option when asked.
2. Install **Git** if it is not there (`git --version` in a terminal to check).
3. Open a terminal (Mac: Terminal; Windows: PowerShell) and run:

   ```
   git clone https://github.com/cvat-ai/cvat
   cd cvat
   docker compose up -d
   ```

   The first start downloads several GB and can take 10-20 minutes.
4. Create your account (once):

   ```
   docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
   ```

   Choose a username (your first name), any email, and a password you will remember.

### 3.2 Start and stop CVAT

- Start: open Docker Desktop, then in a terminal in the `cvat` folder run
  `docker compose up -d`.
- Open **Google Chrome** at <http://localhost:8080> and log in. (CVAT works best in Chrome.)
- Stop when you are done for the day: `docker compose stop`. Your work is kept.
- **Never run `docker compose down -v`**: the `-v` deletes all your annotations.

### 3.3 Create a task for a batch

Do this once per batch, with that batch's unzipped folder.

1. In CVAT, click **Tasks** at the top, then the blue **+** button → **Create new task**.
2. **Name**: the batch name, exactly like the zip: `Mira_batch1` or `Emma_batch1`.
3. **Labels**: click the **Raw** tab. Delete what is in the box, open `labels.json` in a
   text editor, copy all of it, paste it into the box and click **Done**. You should see
   one label, **object**, with two attributes: **name** (text) and **format** (a list).
4. **Select files**: under **My computer**, drag in **all files inside `images/`** (not
   the folder itself, not `annotations.zip` or `labels.json`).
5. Open **Advanced configuration** and set:
   - **Image quality: 100** (the default 70 blurs small objects).
   - **Segment size: 50** (splits the batch into jobs of 50 frames).
   - Leave everything else as it is.
6. Click **Submit & Open** and wait until the task page shows its jobs.

### 3.4 Load the pre-drawn boxes (once per task, before you start)

1. On the task page click **Actions** → **Upload annotations**.
2. **Import format**: **CVAT 1.1**.
3. Drop in `annotations.zip` from the batch folder and click **OK**.
4. Open the first job and check that boxes appear on the frames, and that clicking a box
   shows its name (e.g. `cup`) and a format (e.g. `real`) in the right sidebar.

> **Do this only once, before you start annotating.** Uploading annotations again replaces
> everything in the task, including your corrections. If something looks wrong after
> uploading, stop and ask Jane.

---

## 4. Annotating in CVAT

Open a job from the task page (**Job #…**). Frames are in a fixed order; work through them
in order and finish one job before starting the next.

### 4.1 The screen

- **Left toolbar**: drawing tools. The rectangle tool (**Draw new rectangle**) is the only
  one you need. Click it (the label is always **object**), click **Shape**, then click two
  opposite corners of the object.
- **Right sidebar (Objects)**: every box on the frame. Click an entry to select the box;
  click the small arrow to show its **name** and **format**. Type the name into the name
  field; choose the format from its list.
- **Top bar**: frame number with arrows, **Save**, and the workspace menu (Standard /
  Attribute annotation).

### 4.2 Useful keys

| Key | Action |
|---|---|
| **F** / **D** | next / previous frame |
| **N** | draw another box (press N, then click two corners) |
| **Delete** | delete the selected box |
| **Ctrl+S** (Mac: Cmd+S) | save (do it often) |
| **Ctrl+Z** | undo |
| **Esc** | cancel drawing |
| **F1** | show all keyboard shortcuts |

Drag a box's corner or edge to resize it; drag its middle to move it. Scroll to zoom in
on small objects.

### 4.3 Names and formats quickly: Attribute annotation mode

After the boxes on a frame are right, switch the workspace (top right) from **Standard**
to **Attribute annotation**. CVAT then shows one box at a time, zoomed in, with its
attributes:

- **Up** / **Down** arrows switch between the **name** and **format** attributes,
- on **name**, type or correct the name,
- on **format**, press the number key of the format (shown next to each value),
- **Tab** goes to the next box, **Shift+Tab** to the previous one,
- **F** / **D** still change frames.

This is also the easiest way to check the pre-filled names and formats: Tab through
every box and read its name and format.

Switch back to **Standard** to draw or delete boxes.

### 4.4 Finishing a job

1. Check that no box still has format **unset** or an empty name (see 6.4).
2. **Save**.
3. **Menu** (top left) → **Change job state** → **completed**.

---

## 5. The annotation rules

Work frame by frame in this order: **check every box → add missing objects → check every
name → check every format → save.**

### 5.1 Which objects

Box **every discrete object** you can recognise: anything that is one separate thing with
its own edges, that you could point to and name. That includes furniture and appliances
(cabinet, shelf, dresser, microwave), fixtures and decorations (light, ceiling fan,
curtain, mirror, picture frame, rug), containers and bags, toys, clothing, food, plants,
vehicles, people and body parts (5.6).

Do **not** box:

- **surfaces and places**: walls, floors, ceilings, sky, ground, roads, a room or yard as a
  whole, wall-to-wall carpet (a rug or play mat is an object: box it);
- **parts of an object**: a handle, knob, wheel, lid, button, zipper, label or sticker on
  something, a door frame. Exceptions: body parts (5.6), doors, drawers and windows are
  boxed;
- reflections in mirrors, windows or shiny surfaces;
- anything you cannot recognise (too blurry, dark or small to tell what it is).

### 5.2 Check every pre-drawn box

The boxes come from two different computer models, and they make mistakes. For each box:

| If the box… | Do this |
|---|---|
| is on an object with the right name | keep it; tighten the box if needed (5.4) |
| is on an object with the **wrong name**, or a name that breaks 5.7 | correct the name |
| is on a surface, a part of an object (5.1), or on nothing / background | delete it |
| is on an object you **cannot recognise** (too blurry, dark or small to tell) | delete it |
| is a **second box on the same object** | keep the better one, delete the other |
| covers **several objects** (e.g. two cups in one box) | fix it to one object and draw boxes for the others |

### 5.3 Add what is missing

Look over the whole frame, including edges and dark corners, and box every object (5.1)
that has no box, **however small or cut off by the frame edge**, as long as you can
recognise it. There is **no limit** on the number of boxes per frame.

- **One box per object.** A pair of shoes is two shoe boxes; two hands are two hand boxes.
- **Many of the same thing** (books on a shelf, blocks in a bin, crayons): box each one
  you can tell apart, however many there are.
- **A pile you cannot split** into single items (cereal in a bowl, a heap of crumbs, a
  bunch of grapes): one box around the pile, with the usual name (`cereal`, `grape`).
- **Partly hidden objects**: box only the visible part (5.4).

### 5.4 Drawing boxes

- The box should fit the **visible part** of the object tightly, touching its outermost
  visible pixels on all four sides.
- Do not extend the box to where you guess a hidden part would be.
- If an object is cut off by the frame edge, the box ends at the frame edge.

### 5.5 Format (every box)

Most pre-drawn boxes already have a format. It is a computer's guess and is often wrong,
especially for toys and pictures: check it on every box and change it when it is wrong.
A few pre-drawn boxes and every box you draw start as **unset**; give those a format.

| Format | Use for | Examples |
|---|---|---|
| **real** | the object itself | a real cup, a real dog, a person in the room, a book (the book itself is a real book) |
| **toy** | a 3-D toy or model of the object | stuffed dog, toy car, plastic play food, a doll, toy kitchen oven |
| **picture** | a 2-D depiction on paper, cloth, a wall or packaging | animal in a picture book, photo on the fridge, drawing, poster, print on a shirt, cereal-box picture |
| **screen** | an image shown on a screen | a cat on a phone, a cartoon on TV, a person on a video call |

**The name says what it shows, the format says how it is shown.** A toy car is name
**car**, format **toy**. A drawing of a dog in a book is name **dog**, format **picture**
(and the book itself gets its own **book** box, format **real**). A person on TV is
**person**, format **screen**. The TV itself is a **tv** box, format **real**.

Special cases:

- **Toys that are not a model of anything** (blocks, rattle, stacking rings, shape
  sorter, play mat): name the toy itself (`blocks`, `rattle`), format **toy**. A stuffed
  dog is `dog` + toy; a teddy bear is `teddy bear` + toy.
- **doll**: name `doll`, format **toy**.
- **Inside pictures and screens**, box the things that are shown as objects (characters,
  people, animals, vehicles, food…), not the background scenery. A picture whose content
  you cannot name (a pattern, abstract art) gets one box named `picture`, format
  **picture**.
- **ball, balloon, book**: these are the real objects, so format **real**. A picture of a
  ball in a book is ball + picture.
- If you cannot tell real from toy (a very realistic toy apple), choose the more likely
  one; if you really cannot decide, leave it **real** and add the frame to your questions
  list.

### 5.6 People and body parts

- **person**: box each person other than the child wearing the camera, around the
  visible part of their body, even if they are partly out of the frame.
- If **only one body part** of someone is visible (an adult's hand reaching in from the
  edge), box only that body part, not a person.
- **Body parts**: box the body parts in the word list (section 7: hand, finger, arm,
  foot, toe, leg, knee, ankle, head, face, hair, eye, ear, nose, mouth, lip, tongue,
  tooth, cheek, chin, shoulder, tummy) on everyone, including the child's own hands and
  feet. A body part box can sit inside a person box. Do not box other body parts.
- Small parts: box an eye, nose, mouth or tooth only if you can clearly see it.

### 5.7 How to write a name

- **Lowercase, singular, no description**: `cup`, not `Cup`, `cups` or `red cup`. The only
  plural names are things that are always plural: `glasses`, `pants`, `jeans`, `shorts`,
  `pajamas`, `scissors`, `blocks` (a set), `tights`.
- **Use the word-list word when it fits** (section 7): `couch` not `sofa`, `refrigerator`
  not `fridge`, `tv` not `television`, `telephone` for any phone, `cat` not `kitty`,
  `dog` not `puppy`, `cup` for a mug.
- **Use a more specific name only for a different kind of thing that people call by its
  own name**: `sippy cup`, `high chair`, `ceiling fan`, `cutting board`, `water bottle`,
  `teddy bear`. Not `plastic cup` or `big chair`.
- **Not on the word list**: use the everyday name a parent would use (`cabinet`, `shelf`,
  `rug`, `curtain`, `mirror`, `bag`, `microwave`, `laptop`, `pot`, `marker`, `playpen`).
- **Spell carefully**: names are matched to words by computer, so a typo turns an object
  into a new, unknown word.
- If you do not know the name of something you can see clearly, name it as best you can
  and add the file name to your questions list.

### 5.8 Words that are easy to mix up

| Name | Means | Not |
|---|---|---|
| **glass** | a drinking glass | window glass, eyeglasses |
| **glasses** | eyeglasses, sunglasses | a drinking glass |
| **cup** | a cup or mug, sippy cup | a glass or bowl |
| **dish** | a serving dish or a dish of food you cannot call plate or bowl | use **plate** or **bowl** when it is one |
| **nail** | a metal nail (hardware) | a fingernail (part of **finger**, not boxed) |
| **can** | a metal can (food, drink) | a jar |
| **paper** | sheets of paper, newspaper | napkins (**napkin**), books (**book**), a paper bag (**bag**) |
| **closet** | a closet or wardrobe (doors closed or open) | a chest of drawers (**dresser**, plus **drawer** for each drawer) |
| **plant** | a potted or garden plant | **flower** (the flower itself), **tree**, **grass** |
| **fish** | a fish (animal or food) | |
| **pony** | a pony or horse | |

When in doubt about a name, note the file name in your questions list (6.3).

---

## 6. Saving, exporting and handing in

### 6.1 Save often

Press **Ctrl+S** every few frames. CVAT also autosaves, but do not rely on it.

### 6.2 Export at the end of every session (backup) and when a batch is done

1. Go to the **task** page (not the job).
2. **Actions** → **Export task dataset**.
3. **Export format**: **CVAT for images 1.1**.
4. **Save images**: **off**.
5. **Custom name**: `Mira_batch1_2026-10-21` (your name, batch, today's date).
6. Click **OK**. CVAT prepares the file; download it from the notification or from the
   **Requests** page at the top, when it says Finished.
7. Put the zip in **your folder in the lab's IRB-approved storage**. When a batch is
   finished, name the final export `Mira_batch1_final.zip`.

The export contains only box coordinates and labels, not images.

### 6.3 Log and questions

In your folder, keep two simple notes:

- **Time log**: date, batch, start and end time of each session, and frames done. This
  tells us how long the rest will take.
- **Questions list**: file name + question (e.g. `04529aebb0a3.jpg`: is the red thing on
  the counter a toy or real?). Send it to Jane at the end of each session.

### 6.4 Before you call a batch finished

- Every frame visited (CVAT shows the frame count per job).
- No box with format **unset**: in each job, open **Filters** at the top, add a rule for
  attribute **format** equals **unset**, and apply it; the list should be empty. Clear
  the filter afterwards. (Boxes you draw start as **unset**, so this also finds boxes you
  drew but did not finish; check that those have a name.)
- Every job set to **completed**, then the final export (6.2).

---

## 7. Word list (use these words when they fit)

This is a list of words young children learn (the CDI). Use a word from it whenever it
fits the object (5.7); for anything else, use the everyday name. Objects that are not on
the list are boxed too.

| Group | Words |
|---|---|
| people | person |
| animals | alligator, ant, bear, bee, bird, bunny, butterfly, cat, cow, deer, dog, donkey, duck, elephant, frog, giraffe, goose, hen, horse, lamb, lion, monkey, moose, mouse, owl, penguin, pig, pony, rooster, sheep, squirrel, tiger, turkey, turtle, wolf, zebra |
| body parts | ankle, arm, cheek, chin, ear, eye, face, finger, foot, hair, hand, head, knee, leg, lip, mouth, nose, shoulder, toe, tongue, tooth, tummy |
| clothing | beads, belt, bib, boots, coat, diaper, dress, glove, hat, jacket, jeans, mitten, necklace, pajamas, pants, scarf, shirt, shoe, shorts, slipper, sneaker, snowsuit, sock, sweater, tights, underpants |
| food and drink | apple, applesauce, banana, beans, bread, butter, cake, candy, carrot, cereal, cheerios, cheese, chicken, chips, chocolate, coffee, coke, cookie, corn, cracker, egg, fish, grapes, gum, hamburger, jello, jelly, juice, lollipop, meat, melon, milk, muffin, nut, orange, pancake, pasta, pastry, peas, pickle, pizza, popcorn, popsicle, potato, pretzel, pudding, pumpkin, raisin, sandwich, soda, soup, strawberry, toast, tuna, vitamin, yogurt |
| furniture and rooms | bathtub, bed, bench, chair, closet, couch, crib, door, drawer, dryer, oven, potty, refrigerator, shower, sink, stairs, stove, table, tv, window |
| household | basket, blanket, bottle, bowl, box, broom, brush, bucket, camera, can, clock, comb, cup, dish, fork, glass, glasses, hammer, jar, key, knife, lamp, light, medicine, mop, nail, napkin, paper, pen, penny, picture, pillow, plant, plate, purse, radio, scissors, soap, spoon, tape, telephone, tissue, toothbrush, towel, trash, tray, vacuum, walker, watch |
| outside | cloud, flag, flower, grass, hose, ladder, moon, pool, rock, roof, sandbox, shovel, slide, snowman, sprinkler, star, stick, stone, sun, swing, tree |
| toys | ball, balloon, bat, blocks, book, bubbles, chalk, crayon, doll, glue, pencil, present, puzzle, teddy bear, toy |
| vehicles | airplane, bicycle, boat, bus, car, firetruck, helicopter, motorcycle, sled, stroller, tractor, train, tricycle, truck |

---

## 8. Quick checklist per frame

1. Every pre-drawn box: a discrete object, right name, tight? Otherwise fix or delete.
2. Every discrete object in view has exactly one box (no limit per frame).
3. People boxed; body parts boxed.
4. Every box has a name written as in 5.7, and the right format: real, toy, picture or
   screen (pre-filled formats checked, none left unset).
5. Save.

Questions at any point: ask Jane. When unsure, write down the file name rather than guess
silently.
