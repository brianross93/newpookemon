
If you are still using the old URL (datacrystal.romhacking.net), please update your bookmarks! The old URL may stop working at any time.
The current URL is datacrystal.tcrf.net.
Pokémon Red and Blue/RAM map
< Pokémon Red and Blue
Jump to navigationJump to search
Chip tiny.png The following article is a RAM map for Pokémon Red and Blue.

Currently, many more memory addresses can be found in the Pokémon Red disassembly project (WRAM, VRAM, HRAM, SRAM).

Hmmm...
To do:
Add more addresses that are different in the Japanese version.

Contents
1	Saved data (SRAM)
1.1	Bank 0
1.2	Bank 1
1.3	Bank 2
1.4	Bank 3
2	Main data (WRAM)
2.1	Audio
2.2	Sprite Data
2.3	Tile Data
2.4	Menu Data
2.5	Link Data
2.6	Misc.
2.7	Battle
2.8	Joypad simulation
2.9	Pokémon Mart
2.10	Name Rater
2.11	Battle
2.12	Game Corner
2.13	Link Battle PRNG
2.14	Player
2.15	Pokedex
2.16	Items
2.17	Money
2.18	Rival
2.19	Miscellaneous
2.20	Audio
2.21	Map Header
2.22	Tileset Header
2.23	Tile graphics
2.24	Stored Items
2.25	Game Coins
2.26	Event Flags
2.27	Wild Pokémon
2.28	Opponent Trainer’s Pokémon
2.29	Etc.
2.30	Stored Pokémon
3	Internal data (HRAM)
3.1	OAM DMA
3.2	Misc.
3.3	Serial
3.4	Graphics scroll values
3.5	Joypad
3.6	ROM banking
3.7	VBlank data transfer
3.8	VBlank 2x2 tile block redraw
3.9	RNG
3.10	VBlank interface
3.11	Misc. 2
Saved data (SRAM)
Bank 0
Sprite decompression zone + Hall of Fame

A000 - A187 : Sprite buffer #0
A188 - A30F : Sprite buffer #1
A310 - A497 : Sprite buffer #2
 A498 - A597 : Unused ?
 A598 - B857 : Hall Of Fame data
 B858 - BFFF : Unused ?
Note : this is the reason Missingno corrupts the HoF data : its invalid sprite, when decompressed, overwrites data past the buffers' boundaries... right into HoF data.

Other Pokémon with even larger corruptions reach WRAM - usually leading to crashes.

Bank 1
A000 - A597 : Unused ?
 A598 - A5A2 : Player name
A5A3 - AD2B : Main data
AD2C - AF2B : Sprite data
AF2C - B0BF : Party data
B0C0 - B521 : Current box data
B522        : Tileset type
B523        : Main data checksum
 B524 - BFFF : Unused ?
Bank 2
Pokémon boxes 1-6

A000 - A461 : Box 1
A462 - A8C3 : Box 2
A8C4 - AD25 : Box 3
AD26 - B187 : Box 4
B188 - B5E9 : Box 5
B5EA - BA4B : Box 6
BA4C        : Global checksum
BA4D - BA52 : Individual checksums
 BA53 - BFFF : Unused
Bank 3
Pokémon boxes 7-12

A000 - A461 : Box 7
A462 - A8C3 : Box 8
A8C4 - AD25 : Box 9
AD26 - B187 : Box 10
B188 - B5E9 : Box 11
B5EA - BA4B : Box 12
BA4C        : Global checksum
BA4D - BA52 : Individual checksums
 BA53 - BFFF : Unused
Main data (WRAM)
Audio
Addresses C000 to C0F2 are used by the sound engine.

C002 - Bit 7 : 1 if audio is muted. Other bits : if non-zero, pauses music and continues SFX until it ends.
C0DE - Volumes for all music channels (and fade for those who support it)
C0EF - Current sound bank
C0F0 - Saved sound bank
Bytes C0F3 to C0FF are undocumented (if used)

Sprite Data
C100 to C1FF : Data for all sprites on the current map
               Holds info for 16 sprites with $10 bytes each
               Note : player is always sprite 0

Replace x with the sprite ID
    C1x0: picture ID (fixed, loaded at map init)
    C1x1: movement status (0: uninitialized, 1: ready, 2: delayed, 3: moving)
    C1x2: sprite image index (changed on update, $ff if off screen, includes facing direction, progress in walking animation and a sprite-specific offset)
    C1x3: Y screen position delta (-1,0 or 1; added to c1x4 on each walking animation update)
    C1x4: Y screen position (in pixels, always 4 pixels above grid which makes sprites appear to be in the center of a tile)
    C1x5: X screen position delta (-1,0 or 1; added to c1x6 on each walking animation update)
    C1x6: X screen position (in pixels, snaps to grid if not currently walking)
    C1x7: intra-animation-frame counter (counting upwards to 4 until c1x8 is incremented)
    C1x8: animation frame counter (increased every 4 updates, hold four states (totalling to 16 walking frames)
    C1x9: facing direction (0: down, 4: up, 8: left, $c: right)
    C1xA to C1xF are unudocumented (if used)
 C200 to C2FF : More data for all sprites on the current map
               Holds info for 16 sprites with $10 bytes each
               Note : player sprite is always sprite 0

    C2x0: walk animation counter (counting from $10 backwards when moving)
    C2x1:
    C2x2: Y displacement (initialized at 8, supposed to keep moving sprites from moving too far, but bugged)
    C2x3: X displacement (initialized at 8, supposed to keep moving sprites from moving too far, but bugged)
    C2x4: Y position (in 2x2 tile grid steps, topmost 2x2 tile has value 4)
    C2x5: X position (in 2x2 tile grid steps, leftmost 2x2 tile has value 4)
    C2x6: movement byte 1 (determines whether a sprite can move, $ff:not moving, $fe:random movements, others unknown)
    C2x7: (?) (set to $80 when in grass, else $0; may be used to draw grass above the sprite)
    C2x8: delay until next movement (counted downwards, status (c1x1) is set to ready if reached 0)
    C2x9 to C2xD are undocumented
    C2xE: sprite image base offset (in video ram, player always has value 1, used to compute c1x2)
    C2xF isn't documented
C300 to C39F : OAM DMA buffer (source for the DMA Transfer)

Tile Data
C3A0 to C507 : Buffer of all tiles onscreen
C508 to C5CF : Copy of previous buffer (used to restore tiles after closing menus)
Menu Data
Coordinates of the position of the cursor for the top menu item (id 0)
CC24 : Y position
CC25 : X position
 CC26 - Currently selected menu item (topmost is 0)
CC27 - Tile "hidden" by the menu cursor
CC28 - ID of the last menu item
CC29 - bitmask applied to the key port for the current menu
CC2A - ID of the previously selected menu item
CC2B - Last position of the cursor on the party / Bill's PC screen
CC2C - Last position of the cursor on the item screen
CC2D - Last position of the cursor on the START / battle menu
CC2F - Index (in party) of the Pokémon currently sent out
CC30~CC31 - Pointer to cursor tile in C3A0 buffer
CC36 - ID of the first displayed menu item
CC35 - Item highlighted with Select (01 = first item, 00 = no item, etc.)
CC3A and CC3B are unused

Link Data
CC3C to CC49 hold data used for Cable Club stuff

The meaning of CC47 depends on the context
CC47 - Link timeout counter
CC47 - Is player entering Cable Club ?
Misc.
CC97 to CCA0 : buffer for when swapping party Pokémon
Battle
CCD5 - Number of turns in current battle
CCD6 is undocumented (if used)
CCD7 - Player's Substitute HP
CCD8 - Enemy Substitute HP
CCDB - Move menu type : 0 is regular, 1 is mimic, other are text boxes (learn, PP-refill...)
CCDC - Player-selected move
CCDD - Enemy-selected move
CCE5~CCE7 - Money earned by Pay Day
Safari Zone data :

CCE8 - Opponent escaping factor
CCE9 - Opponent baiting factor
 CCED - Is current Pokémon disobedient ?
CCEE - Player move that the enemy disabled
CCEF - Enemy move that the player disabled
CCF6 - Is low-health alarm disabled ?
CD05~CD06 - Amount of damage the enemy accumulated while Biding
CD1A - Player's Pokémon Attack modifier (7 means no modifier)
CD1B - Player's Pokémon Defense modifier
CD1C - Player's Pokémon Speed modifier
CD1D - Player's Pokémon Special modifier
CD1E - Player's Pokémon Accuracy modifier
CD1F - Player's Pokémon Evasion modifier
CD2D - Engaged Trainer class / legendary Pokémon ID
CD2E - Engaged Trainer roster ID / Enemy's Pokémon Attack modifier (7 means no modifier)
CD2F - Enemy's Pokémon Defense modifier
CD30 - Enemy's Pokémon Speed modifier
CD31 - Enemy's Pokémon Special modifier
CD32 - Enemy's Pokémon Accuracy modifier
CD33 - Enemy's Pokémon Evasion modifier
Joypad simulation
CD38 - Index for joypad input simulation. If non-zero, disables collision but does not lock player.
Pokémon Mart
JPN addr.	INT addr.	Description
CF62	CF7B	Total Items
CF63	CF7C	Item 1
CF64	CF7D	Item 2
CF65	CF7E	Item 3
CF66	CF7F	Item 4
CF67	CF80	Item 5
CF68	CF81	Item 6
CF69	CF82	Item 7
CF70	CF83	Item 8
CF71	CF84	Item 9
CF72	CF85	Item 10
Name Rater
CF92 - Which Pokémon does Name Rater Change?
Battle
CFD3 - Your Move Effect (e.g. 10h = coins scatter everywhere)
CFD5 - Your Move Type
CCDC - Your Move Used
JPN addr.	INT addr.	Description
CFB3	CFCC	Enemy's Move ID
CFB4	CFCD	Enemy's Move Effect
CFB5	CFCE	Enemy's Move Power
CFB6	CFCF	Enemy's Move Type
CFB7	CFD0	Enemy's Move Accuracy
CFB8	CFD1	Enemy's Move Max PP
CFB9	CFD2	Player's Move ID
CFBA	CFD3	Player's Move Effect
CFBB	CFD4	Player's Move Power
CFBC	CFD5	Player's Move Type
CFBD	CFD6	Player's Move Accuracy
CFBE	CFD7	Player's Move Max PP
CFBF	CFD8	Enemy's Pokémon internal ID
CFC0	CFD9	Player's Pokémon internal ID
CFC1-CFCB	CFDA-CFE4	Enemy's Name
CFCC	CFE5	Enemy's Pokémon internal ID
CFCD-CFCE	CFE6-CFE7	Enemy's HP
CFCF	CFE8	Enemy's Level
CFD0	CFE9	Enemy's Status
Bit 6 : Paralyzed
Bit 5 : Frozen
Bit 4 : Burned
Bit 3 : Poisoned
Bits 0-2 : Sleep counter
CFD1	CFEA	Enemy's Type 1
CFD2	CFEB	Enemy's Type 2
CFD3	CFEC	Enemy's Catch Rate (Unused, only referenced by Transform script, the one actually used is at CFEE (JPN)/D007 (INT))
CFD4	CFED	Enemy's Move 1
CFD5	CFEE	Enemy's Move 2
CFD6	CFEF	Enemy's Move 3
CFD7	CFF0	Enemy's Move 4
CFD8	CFF1	Enemy's Attack and Defense IVs
CFD9	CFF2	Enemy's Speed and Special IVs
CFDA	CFF3	Enemy's Level
CFDB-CFDC	CFF4-CFF5	Enemy's Max HP
CFDD-CFDE	CFF6-CFF7	Enemy's Attack
CFDF-CFE0	CFF8-CFF9	Enemy's Defense
CFE1-CFE2	CFFA-CFFB	Enemy's Speed
CFE3-CFE4	CFFC-CFFD	Enemy's Special
CFE5	CFFE	Enemy's PP (First Slot)
CFE6	CFFF	Enemy's PP (Second Slot)
CFE7	D000	Enemy's PP (Third Slot)
CFE8	D001	Enemy's PP (Fourth Slot)
CFE9-CFED	D002-D006	Enemy's Base Stats
CFEE	D007	Enemy's Catch Rate
CFEF	D008	Enemy's Base Experience
Pokémon 1st Slot (In-Battle)

JPN addr.	INT addr.	Description
CFF0-CFFA	D009-D013	Name
CFFB	D014	Pokémon Number
CFFC-CFFD	D015-D016	Current HP
CFFE	D017	
CFFF	D018	Status
Bit 6 : Paralyzed
Bit 5 : Frozen
Bit 4 : Burned
Bit 3 : Poisoned
Bits 0-2 : Sleep counter
D000	D019	Type 1
D001	D01A	Type 2
D002	D01B	
D003	D01C	Move #1 (First Slot)
D004	D01D	Move #2 (Second Slot)
D005	D01E	Move #3 (Third Slot)
D006	D01F	Move #4 (Fourth Slot)
D007	D020	Attack and Defense DVs
D008	D021	Speed and Special DVs
D009	D022	Level
D00A-D00B	D023-D024	Max HP
D00C-D00D	D025-D026	Attack
D00E-D00F	D027-D028	Defense
D010-D011	D029-D02A	Speed
D012-D013	D02B-D02C	Special
D014	D02D	PP (First Slot)
D015	D02E	PP (Second Slot)
D016	D02F	PP (Third Slot)
D017	D030	PP (Fourth Slot)
D057 - Type of battle
 D05A - Battle Type (Normal battle, Safari Zone, Old Man battle...)
 D05C - Is Gym Leader battle music playing?
D05D - 
D05E - Critical Hit / OHKO Flag
       01 - Critical Hit!
       02 - One-hit KO!
D05F - Hooked Pokémon Flag
 D062-D064 - Battle Status (Player)
          D062:
            bit 0 - Bide
            bit 1 - Thrash / petal dance
            bit 2 - Attacking multiple times (e.g. double kick)
            bit 3 - Flinch
            bit 4 - Charging up for attack
            bit 5 - Using multi-turn move (e.g. wrap)
            bit 6 - Invulnerable to normal attack (using fly/dig)
            bit 7 - Confusion
          D063:
            bit 0 - X Accuracy effect
            bit 1 - protected by "mist"
            bit 2 - focus energy effect
            bit 4 - has a substitute
            bit 5 - need to recharge
            bit 6 - rage
            bit 7 - leech seeded
          D064:
            bit 0 - toxic
            bit 1 - light screen
            bit 2 - reflect
            bit 3 - tranformed
D065 - Stat to double (CPU)
D066 - Stat to halve (CPU)
D067-D069 - Battle Status (CPU) - Includes the "Transformed" status in D069 which makes the game regard the opponent as a Ditto.
D06A - Multi-Hit Move counter (Player)
D06B - Confusion counter (Player)
D06C - Toxic counter (Player)
D06D-D06E - Disable counter (Player)
D06F - Multi-Hit Move counter (CPU)
D070 - Confusion counter (CPU)
D071 - Toxic counter (CPU)
D072 - Disable counter (CPU)
 D0D8 - Amount of damage attack is about to do. Max possible damage may appear one frame before actual damage.
Game Corner
D13D - 1st Game Corner Prize
D13E - 2nd Game Corner Prize
D13F - 3rd Game Corner Prize
Link Battle PRNG
D148-D150 - 9 Pseudo-Random Numbers used during Link Battles. Once a batch is used up, it generates a new batch with n × 5 + 1.
Player
D158-D162 - Your Name
 D163 - # Pokémon In Party
D164 - Pokémon 1
D165 - Pokémon 2
D166 - Pokémon 3
D167 - Pokémon 4
D168 - Pokémon 5
D169 - Pokémon 6
D16A - End of list
Pokémon 1

D16B - Pokémon (Again)
D16C-D16D - Current HP
D16E - 'Level' (not the actual level, see the notes article)
D16F - Status (Poisoned, Paralyzed, etc.)
D170 - Type 1
D171 - Type 2
D172 - Catch rate/Held item (When traded to Generation II)
D173 - Move 1
D174 - Move 2
D175 - Move 3
D176 - Move 4
D177-D178 - Trainer ID
D179-D17B - Experience
D17C-D17D - HP EV
D17E-D17F - Attack EV
D180-D181 - Defense EV
D182-D183 - Speed EV
D184-D185 - Special EV
D186 - Attack/Defense IV
D187 - Speed/Special IV
D188 - PP Move 1
D189 - PP Move 2
D18A - PP Move 3
D18B - PP Move 4
D18C - Level (actual level)
D18D-D18E - Max HP
D18F-D190 - Attack
D191-D192 - Defense
D193-D194 - Speed
D195-D196 - Special
Pokémon 2

D197 - Pokémon
D198-D199 - Current HP
D19A - 'Level' (not the actual level, see the notes article)
D19B - Status
D19C - Type 1
D19D - Type 2
D19E - Catch rate/Held item (When traded to Generation II)
D19F - Move 1
D1A0 - Move 2
D1A1 - Move 3
D1A2 - Move 4
D1A3-D1A4 - Trainer ID
D1A5-D1A7 - Experience 
D1A8-D1A9 - HP EV
D1AA-D1AB - Attack EV
D1AC-D1AD - Defense EV
D1AE-D1AF - Speed EV
D1B0-D1B1 - Special EV
D1B2 - Attack/Defense IV
D1B3 - Speed/Special IV
D1B4 - PP Move 1
D1B5 - PP Move 2
D1B6 - PP Move 3
D1B7 - PP Move 4
D1B8 - Level (actual)
D1B9-D1BA - Max HP
D1BB-D1BC - Attack
D1BD-D1BE - Defense
D1BF-D1C0 - Speed
D1C1-D1C2 - Special
Pokémon 3

D1C3 - Pokémon
D1C4-D1C5 - Current HP
D1C6 - 'Level' (not the actual level, see the notes article)
D1C7 - Status
D1C8 - Type 1
D1C9 - Type 2
D1CA - Catch rate/Held item (When traded to Generation II)
D1CB - Move 1
D1CC - Move 2
D1CD - Move 3
D1CE - Move 4
D1CF-D1D0 - Trainer ID
D1D1-D1D3 - Experience
D1D4-D1D5 - HP EV
D1D6-D1D7 - Attack EV
D1D8-D1D9 - Defense EV
D1DA-D1DB - Speed EV
D1DC-D1DD - Special EV
D1DE - Attack/Defense IV
D1DF - Speed/Special IV
D1E0 - PP Move 1
D1E1 - PP Move 2
D1E2 - PP Move 3
D1E3 - PP Move 4
D1E4 - Level
D1E5-D1E6 - Max HP
D1E7-D1E8 - Attack
D1E9-D1EA - Defense
D1EB-D1EC - Speed
D1ED-D1EE - Special
Pokémon 4

D1EF - Pokémon
D1F0-D1F1 - Current HP
D1F2 - 'Level' (not the actual level, see the notes article)
D1F3 - Status
D1F4 - Type 1
D1F5 - Type 2
D1F6 - Catch rate/Held item (When traded to Generation II)
D1F7 - Move 1
D1F8 - Move 2
D1F9 - Move 3
D1FA - Move 4
D1FB-D1FC - Trainer ID
D1FD-D1FF - Experience 
D200-D201 - HP EV
D202-D203 - Attack EV
D204-D205 - Defense EV
D206-D207 - Speed EV
D208-D209 - Special EV
D20A - Attack/Defense IV
D20B - Speed/Special IV
D20C - PP Move 1
D20D - PP Move 2
D20E - PP Move 3
D20F - PP Move 4
D210 - Level
D211-D212 - Max HP
D213-D214 - Attack
D215-D216 - Defense
D217-D218 - Speed
D219-D21A - Special
Pokémon 5

D21B - Pokémon
D21C-D21D - Current HP
D21E - 'Level' (not the actual level, see the notes article)
D21F - Status
D220 - Type 1
D221 - Type 2
D222 - Catch rate/Held item (When traded to Generation II)
D223 - Move 1
D224 - Move 2
D225 - Move 3
D226 - Move 4
D227-D228 - Trainer ID
D229-D22B - Experience 
D22C-D22D - HP EV
D22E-D22F - Attack EV
D230-D231 - Defense EV
D232-D233 - Speed EV
D234-D235 - Special EV
D236 - Attack/Defense IV
D237 - Speed/Special IV
D238 - PP Move 1
D239 - PP Move 2
D23A - PP Move 3
D23B - PP Move 4
D23C - Level
D23D-D23E - Max HP
D23F-D240 - Attack
D241-D242 - Defense
D243-D244 - Speed
D245-D246 - Special
Pokémon 6

D247 - Pokémon
D248-D249 - Current HP
D24A - 'Level' (not the actual level, see the notes article)
D24B - Status
D24C - Type 1
D24D - Type 2
D24E - Catch rate/Held item (When traded to Generation II)
D24F - Move 1
D250 - Move 2
D251 - Move 3
D252 - Move 4
D253-D254 - Trainer ID
D255-D257 - Experience 
D258-D259 - HP EV
D25A-D25B - Attack EV
D25C-D25D - Defense EV
D25E-D25F - Speed EV
D260-D261 - Special EV
D262 - Attack/Defense IV
D263 - Speed/Special IV
D264 - PP Move 1
D265 - PP Move 2
D266 - PP Move 3
D267 - PP Move 4
D268 - Level
D269-D26A - Max HP
D26B-D26C - Attack
D26D-D26E - Defense
D26F-D270 - Speed
D271-D272 - Special
Trainer Name

D273-D27D - Trainer name for 1st
D27E-D288 - Trainer name for 2nd
D289-D293 - Trainer name for 3rd
D294-D29E - Trainer name for 4th
D29F-D2A9 - Trainer name for 5th
D2AA-D2B4 - Trainer name for 6th
Nickname

D2B5-D2BF - Nickname for 1st
D2C0-D2CA - Nickname for 2nd
D2CB-D2D5 - Nickname for 3rd
D2D6-D2E0 - Nickname for 4th
D2E1-D2EB - Nickname for 5th
D2EC-D2F6 - Nickname for 6th
Pokedex
JPN addr.	INT addr.	Description
D27B	D2F7	Own 1-8
D27C	D2F8	Own 9-16
D27D	D2F9	Own 17-24
D27E	D2FA	Own 25-32
D27F	D2FB	Own 33-40
D280	D2FC	Own 41-48
D281	D2FD	Own 49-56
D282	D2FE	Own 57-64
D283	D2FF	Own 65-72
D284	D300	Own 73-80
D285	D301	Own 81-88
D286	D302	Own 89-96
D287	D303	Own 97-104
D288	D304	Own 105-112
D289	D305	Own 113-120
D28A	D306	Own 121-128
D28B	D307	Own 129-136
D28C	D308	Own 137-144
D28D	D309	Own 145-152
D28E	D30A	Seen 1-8
D28F	D30B	Seen 9-16
D290	D30C	Seen 17-24
D291	D30D	Seen 25-32
D292	D30E	Seen 33-40
D293	D30F	Seen 41-48
D294	D310	Seen 49-56
D295	D311	Seen 57-64
D296	D312	Seen 65-72
D297	D313	Seen 73-80
D298	D314	Seen 81-88
D299	D315	Seen 89-96
D29A	D316	Seen 97-104
D29B	D317	Seen 105-112
D29C	D318	Seen 113-120
D29D	D319	Seen 121-128
D29E	D31A	Seen 129-136
D29F	D31B	Seen 137-144
D2A0	D31C	Seen 145-152
Items
D31D - Total Items
D31E - Item 1
D31F - Item 1 Quantity
D320 - Item 2
D321 - Item 2 Quantity
D322 - Item 3
D323 - Item 3 Quantity
D324 - Item 4
D325 - Item 4 Quantity
D326 - Item 5
D327 - Item 5 Quantity
D328 - Item 6
D329 - Item 6 Quantity
D32A - Item 7
D32B - Item 7 Quantity
D32C - Item 8
D32D - Item 8 Quantity
D32E - Item 9
D32F - Item 9 Quantity
D330 - Item 10
D331 - Item 10 Quantity
D332 - Item 11
D333 - Item 11 Quantity
D334 - Item 12
D335 - Item 12 Quantity
D336 - Item 13
D337 - Item 13 Quantity
D338 - Item 14
D339 - Item 14 Quantity
D33A - Item 15
D33B - Item 15 Quantity
D33C - Item 16
D33D - Item 16 Quantity
D33E - Item 17
D33F - Item 17 Quantity
D340 - Item 18
D341 - Item 18 Quantity
D342 - Item 19
D343 - Item 19 Quantity
D344 - Item 20
D345 - Item 20 Quantity
D346 - Item End of List
Money
D347 - Money Byte 1
D348 - Money Byte 2
D349 - Money Byte 3
Rival
D34A-D351 - Rival's Name
Miscellaneous
D355 - Options

Bit 7 = Battle Animation (1 = Off, 0 = On)
Bit 6 = Battle Style (1 = Set, 0 = Shift)
Bit 5-4 = probably unused
Low Nybble = Text Speed (0x0 = fastest, 0xF = slowest)
Fast = 1
Medium = 3
Slow = 5
D356 = Badges (Binary Switches)

D358 = If bit0 = 0, delay is limited to 1 frame between each letter. If bit1 = 0, no delay during text printing. Overrides previous setting.

D359 = Player ID [1] (Multiple of 256) D35A = Player ID [2] (...+1-255)

D35B = Audio track (See Audio section) D35C = Audio bank (See Audio section)

D35D = Controls the map's palette. Usually 0, but is set to 6 when Flash is required.

D35E = Current Map Number

D35F-D360 = Event Displacement (view Notes, for more info)
D361 - 1 byte integer = Current Player Y-Position
D362 - 1 byte integer = Current Player X-Position
D363 = Current Player Y-Position (Current Block)
D364 = Current Player X-Position (Current Block)
D365 = Last map location for walking through certain exits (e.g. Safari Zone gate)

Audio
C022 = Audio track channel 1

C023 = Audio track channel 2

C024 = Audio track channel 3

C025 = Audio track channel 4

C0E6 = Current wavetable in use by the wave channel (audio track channel 3)

C0E8, C0E9 = Music tempo (lower values in C0E9 may = faster except for particularly low values)

C0EA, C0EB = Sound effects tempo

C0EC, C0ED = Sound effects header pointer

C0EE = New sound ID

C0EF = Audio bank

C0F0 = Audio bank

D35B = Audio track in current map

D35C = Audio bank in current map

Bank 02
B3 = Validate
B4 = Bump
B5 = Exit of a house
B6 = Save Game
B8 = Pokéflute
B9 = Ding Dong
BA = PalletTown
BD = Pokecenter
C0 = Gym
C3 = Viridian City, Pewter City, Saffron City
C7 = Cerulean City, Fuchsia City
CA = Celadon City
CD = Cinnabar City
D0 = Vermilion City
D4 = Lavender Town
D8 = SS Anne
DB = Meet Oak
DE = Meet Blue
E1 = Follow me
E5 = Safari Zone / Evolution
E8 = Pokémon Healding
EB = Road 1, 2
EF = Road 24, 25
F3 = Road 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22
F7 = Road 11, 12, 13, 14, 15
FB = Indigo Plateau
Bank 08
EA = Gym Leader Battle
ED = Trainer Battle
F0 = Wild Battle
F3 = Final Battle
F6 = Defeated Trainer
F9 = Defeated Wild Pokemon
FC = Defeated Champion / Gym
Bank 1F
C3 = Title Screen
C7 = Credits
CA = Hall of fame
CD = Oak's Lab
D0 = Jigglypuff Song
D2 = Bike Riding
D6 = Surfing
D9 = Came Corner
DC = Intro Game
DF = Power Plant, Unknown Dungeon, Rocket HQ
E3 = Viridian Forest, Seafoam Islands
E7 = Mt. Moon, Rock Tunnel, Victory Road
EB = Cinnabar City Mansion
EF = Pokemon Tower
F2 = Silph Co.
F5 = Eye contact Evil Trainer
F8 = Eye contact Female Trainer
FB = Eye contact Male Trainer
FF = Stop Sound
Map Header
D367 - 1 byte integer = Map's Tileset, 1 byte
D368 - 1 byte integer = Map's Height (Blocks)
D369 - 1 byte integer = Map's Width (Blocks)
D36A-D36B = Map's Data
D36C-D36D = Map's Text Pointer Table (For Each NPC) Location
D36E-D36F = Map's Level-Script Pointer
D370 = Map's Connection Byte
D371-D37B = Map's 1st Connection Data
D37C-D386 = Map's 2nd Connection Data
D387-D391 = Map's 3rd Connection Data
D392-D39C = Map's 4th Connection Data
Tileset Header
D52B = Tileset Bank
D52C-D52D = Pointer to Blocks
D52E-D52F = Pointer to GFX
D530-D531 = Pointer to Collision Data
D532-D534 = "Talking-Over" Tiles
D535 = Grass Tile
Tile graphics
Tiles are 8x8 pixels, stored in 16 bytes, with each byte representing 4 pixels.

For size comparison, the player is 16x16 pixels, therefore, 4 tiles.

"Active" tiles are stored sequentially in RAM, starting at address 0x8000. From address 0x8000 to 0x9000 there are 256 tiles which make up the player in all orientations, and some commonly used stuff. The real map tiles are from 0x9000 to 0x9600.

8000 - 16 byte tile graphic struct = tile 0 (often the top left player tile)
8010 - 16 byte tile graphic struct = tile 1 (often the top right player tile)
8020 - 16 byte tile graphic struct = tile 2 (often the bottom left player tile)
...

9000 - 16 byte tile graphic struct = map tile 0 (often blank)
9010 - 16 byte tile graphic struct = map tile 1
9020 - 16 byte tile graphic struct = map tile 2
...

Stored Items
JPN addr.	INT addr.	Description
D4B9	D53A	Total items in storage
D4BA	D53B	Stored Item 1
D4BB	D53C	Stored Item 1 Quantity
D4BC	D53D	Stored Item 2
D4BD	D53E	Stored Item 2 Quantity
D4BE	D53F	Stored Item 3
D4BF	D540	Stored Item 3 Quantity
D4C0	D541	Stored Item 4
D4C1	D542	Stored Item 4 Quantity
D4C2	D543	Stored Item 5
D4C3	D544	Stored Item 5 Quantity
D4C4	D545	Stored Item 6
D4C5	D546	Stored Item 6 Quantity
D4C6	D547	Stored Item 7
D4C7	D548	Stored Item 7 Quantity
D4C8	D549	Stored Item 8
D4C9	D54A	Stored Item 8 Quantity
D4CA	D54B	Stored Item 9
D4CB	D54C	Stored Item 9 Quantity
D4CC	D54D	Stored Item 10
D4CD	D54E	Stored Item 10 Quantity
D4CE	D54F	Stored Item 11
D4CF	D550	Stored Item 11 Quantity
D4D0	D551	Stored Item 12
D4D1	D552	Stored Item 12 Quantity
D4D2	D553	Stored Item 13
D4D3	D554	Stored Item 13 Quantity
D4D4	D555	Stored Item 14
D4D5	D556	Stored Item 14 Quantity
D4D6	D557	Stored Item 15
D4D7	D558	Stored Item 15 Quantity
D4D8	D559	Stored Item 16
D4D9	D55A	Stored Item 16 Quantity
D4DA	D55B	Stored Item 17
D4DB	D55C	Stored Item 17 Quantity
D4DC	D55D	Stored Item 18
D4DD	D55E	Stored Item 18 Quantity
D4DE	D55F	Stored Item 19
D4DF	D560	Stored Item 19 Quantity
D4E0	D561	Stored Item 20
D4E1	D562	Stored Item 20 Quantity
D4E2	D563	Stored Item 21
D4E3	D564	Stored Item 21 Quantity
D4E4	D565	Stored Item 22
D4E5	D566	Stored Item 22 Quantity
D4E6	D567	Stored Item 23
D4E7	D568	Stored Item 23 Quantity
D4E8	D569	Stored Item 24
D4E9	D56A	Stored Item 24 Quantity
D4EA	D56B	Stored Item 25
D4EB	D56C	Stored Item 25 Quantity
D4EC	D56D	Stored Item 26
D4ED	D56E	Stored Item 26 Quantity
D4EE	D56F	Stored Item 27
D4EF	D570	Stored Item 27 Quantity
D4F0	D571	Stored Item 28
D4F1	D572	Stored Item 28 Quantity
D4F2	D573	Stored Item 29
D4F3	D574	Stored Item 29 Quantity
D4F4	D575	Stored Item 30
D4F5	D576	Stored Item 30 Quantity
D4F6	D577	Stored Item 31
D4F7	D578	Stored Item 31 Quantity
D4F8	D579	Stored Item 32
D4F9	D57A	Stored Item 32 Quantity
D4FA	D57B	Stored Item 33
D4FB	D57C	Stored Item 33 Quantity
D4FC	D57D	Stored Item 34
D4FD	D57E	Stored Item 34 Quantity
D4FE	D57F	Stored Item 35
D4FF	D580	Stored Item 35 Quantity
D500	D581	Stored Item 36
D501	D582	Stored Item 36 Quantity
D502	D583	Stored Item 37
D503	D584	Stored Item 37 Quantity
D504	D585	Stored Item 38
D505	D586	Stored Item 38 Quantity
D506	D587	Stored Item 39
D507	D588	Stored Item 39 Quantity
D508	D589	Stored Item 40
D509	D58A	Stored Item 40 Quantity
D50A	D58B	Stored Item 41
D50B	D58C	Stored Item 41 Quantity
D50C	D58D	Stored Item 42
D50D	D58E	Stored Item 42 Quantity
D50E	D58F	Stored Item 43
D50F	D590	Stored Item 43 Quantity
D510	D591	Stored Item 44
D511	D592	Stored Item 44 Quantity
D512	D593	Stored Item 45
D513	D594	Stored Item 45 Quantity
D514	D595	Stored Item 46
D515	D596	Stored Item 46 Quantity
D516	D597	Stored Item 47
D517	D598	Stored Item 47 Quantity
D518	D599	Stored Item 48
D519	D59A	Stored Item 48 Quantity
D51A	D59B	Stored Item 49
D51B	D59C	Stored Item 49 Quantity
D51C	D59D	Stored Item 50
D51D	D59E	Stored Item 50 Quantity
D51E	D59F	Stored Items End of List
Game Coins
D5A4 - Casino Chips Byte 1
D5A5 - Casino Chips Byte 2
Event Flags
D5A6 to D5C5 : Missable Objects Flags (flags for every (dis)appearing sprites, like the guard in Cerulean City or the Pokéballs in Oak's Lab)

D5AB - Starters Back?

D5C0(bit 1) - 0=Mewtwo appears, 1=Doesn't (See D85F)

D5F3 - Have Town map?

D60D - Have Oak's Parcel?

D700 - Bike Speed

D70B - Fly Anywhere Byte 1
D70C - Fly Anywhere Byte 2
D70D - Safari Zone Time Byte 1
D70E - Safari Zone Time Byte 2
D710 - Fossilized Pokémon?

D714 - Position in Air

D72E - Did you get Lapras Yet?

D732 - Debug New Game

D751 - Fought Giovanni Yet?

D755 - Fought Brock Yet?

D75E - Fought Misty Yet?

D773 - Fought Lt. Surge Yet?

D77C - Fought Erika Yet?

D782 - Fought Articuno Yet?

D790 - If bit 7 is set, Safari Game over

D792 - Fought Koga Yet?

D79A - Fought Blaine Yet?

D7B3 - Fought Sabrina Yet?

D7D4 - Fought Zapdos Yet?

D7D8 - Fought Snorlax Yet (Vermilion)

D7E0 - Fought Snorlax Yet? (Celadon)

D7EE - Fought Moltres Yet?

D803 - Is SS Anne here?

D85F - Mewtwo can be caught if bit 2 clear - Needs D5C0 bit 1 clear, too

Wild Pokémon
D887 - Wild Pokémon encounter rates

Common Battles:

D888 - Level of Encounter 1
D889 - Pokémon Data Value 1
D88A - Level of Encounter 2
D88B - Pokémon Data Value 2
D88C - Level of Encounter 3
D88D - Pokémon Data Value 3
D88E - Level of Encounter 4
D88F - Pokémon Data Value 4
Uncommon Battles:

D890 - Level of Encounter 1
D891 - Pokémon Data Value 1
D892 - Level of Encounter 2
D893 - Pokémon Data Value 2
D894 - Level of Encounter 3
D895 - Pokémon Data Value 3
D896 - Level of Encounter 4
D897 - Pokémon Data Value 4
Rare Battles:

D898 - Level of Encounter 1
D899 - Pokémon Data Value 1
D89A - Level of Encounter 2
D89B - Pokémon Data Value 2
Opponent Trainer’s Pokémon
D89C - Total enemy Pokémon
D89D - Pokémon 1
D89E - Pokémon 2
D89F - Pokémon 3
D8A0 - Pokémon 4
D8A1 - Pokémon 5
D8A2 - Pokémon 6
D8A3 - End of list
Pokémon 1

D8A4 - Pokémon
D8A5-D8A6 - Current HP
D8A7 -
D8A8 - Status
D8A9 - Type 1
D8AA - Type 2
D8AB -
D8AC - Move 1
D8AD - Move 2
D8AE - Move 3
D8AF - Move 4
D8B0-D8B1 - Trainer ID
D8B2-D8B4 - Experience
D8B5-D8B6 - HP EV
D8B7-D8B8 - Attack EV
D8B9-D8BA - Defense EV
D8BB-D8BC - Speed EV
D8BD-D8BE - Special EV
D8BF - Attack/Defense IV
D8C0 - Speed/Special IV
D8C1 - PP Move 1
D8C2 - PP Move 2
D8C3 - PP Move 3
D8C4 - PP Move 4
D8C5 - Level
D8C6-D8C7 - Max HP
D8C8-D8C9 - Attack
D8CA-D8CB - Defense
D8CC-D8CD - Speed
D8CE-D8CF - Special
Pokémon 2

D8D0 - Pokémon
D8D1-D8D2 - Current HP
D8D3 -
D8D4 - Status
D8D5 - Type 1
D8D6 - Type 2
D8D7 -
D8D8 - Move 1
D8D9 - Move 2
D8DA - Move 3
D8DB - Move 4
D8DC-D8DD - Trainer ID
D8DE-D8E0 - Experience
D8E1-D8E2 - HP EV
D8E3-D8E4 - Attack EV
D8E5-D8E6 - Defense EV
D8E7-D8E8 - Speed EV
D8E9-D8EA - Special EV
D8EB - Attack/Defense IV
D8EC - Speed/Special IV
D8ED - PP Move 1
D8EE - PP Move 2
D8EF - PP Move 3
D8F0 - PP Move 4
D8F1 - Level
D8F2-D8F3 - Max HP
D8F4-D8F5 - Attack
D8F6-D8F7 - Defense
D8F8-D8F9 - Speed
D8FA-D8FB - Special
Pokémon 3

D8FC - Pokémon
D8FD-D8FE - Current HP
D8FF -
D900 - Status
D901 - Type 1
D902 - Type 2
D903 -
D904 - Move 1
D905 - Move 2
D906 - Move 3
D907 - Move 4
D908-D909 - Trainer ID
D90A-D90C - Experience
D90D-D90E - HP EV
D90F-D910 - Attack EV
D911-D912 - Defense EV
D913-D914 - Speed EV
D915-D916 - Special EV
D917 - Attack/Defense IV
D918 - Speed/Special IV
D919 - PP Move 1
D91A - PP Move 2
D91B - PP Move 3
D91C - PP Move 4
D91D - Level
D91E-D91F - Max HP
D920-D921 - Attack
D922-D923 - Defense
D924-D925 - Speed
D926-D927 - Special
Pokémon 4

D928 - Pokémon
D929-D92A - Current HP
D92B -
D92C - Status
D92D - Type 1
D92E - Type 2
D92F -
D930 - Move 1
D931 - Move 2
D932 - Move 3
D933 - Move 4
D934-D935 - Trainer ID
D936-D938 - Experience
D939-D93A - HP EV
D93B-D93C - Attack EV
D93D-D93E - Defense EV
D93F-D940 - Speed EV
D941-D942 - Special EV
D943 - Attack/Defense IV
D944 - Speed/Special IV
D945 - PP Move 1
D946 - PP Move 2
D947 - PP Move 3
D948 - PP Move 4
D949 - Level
D94A-D94B - Max HP
D94C-D94D - Attack
D94E-D94F - Defense
D950-D951 - Speed
D952-D953 - Special
Pokémon 5

D954 - Pokémon
D955-D956 - Current HP
D957 -
D958 - Status
D959 - Type 1
D95A - Type 2
D95B -
D95C - Move 1
D95D - Move 2
D95E - Move 3
D95F - Move 4
D960-D961 - Trainer ID
D962-D964 - Experience
D965-D966 - HP EV
D967-D968 - Attack EV
D969-D96A - Defense EV
D96B-D96C - Speed EV
D96D-D96E - Special EV
D96F - Attack/Defense IV
D970 - Speed/Special IV
D971 - PP Move 1
D972 - PP Move 2
D973 - PP Move 3
D974 - PP Move 4
D975 - Level
D976-D977 - Max HP
D978-D979 - Attack
D97A-D97B - Defense
D97C-D97D - Speed
D97E-D97F - Special
Pokémon 6

D980 - Pokémon
D981-D982 - Current HP
D983 -
D984 - Status
D985 - Type 1
D986 - Type 2
D987 -
D988 - Move 1
D989 - Move 2
D98A - Move 3
D98B - Move 4
D98C-D98D - Trainer ID
D98E-D990 - Experience
D991-D992 - HP EV
D993-D994 - Attack EV
D995-D996 - Defense EV
D997-D998 - Speed EV
D999-D99A - Special EV
D99B - Attack/Defense IV
D99C - Speed/Special IV
D99D - PP Move 1
D99E - PP Move 2
D99F - PP Move 3
D9A0 - PP Move 4
D9A1 - Level
D9A2-D9A3 - Max HP
D9A4-D9A5 - Attack
D9A6-D9A7 - Defense
D9A8-D9A9 - Speed
D9AA-D9AB - Special
Trainer Name

D9AC-D9B6 - Trainer name for 1st
D9B7-D9C1 - Trainer name for 2nd
D9C2-D9CC - Trainer name for 3rd
D9CD-D9D7 - Trainer name for 4th
D9D8-D9E2 - Trainer name for 5th
D9E3-D9ED - Trainer name for 6th
Nickname

D9EE-D9F8 - Nickname for 1st
D9F9-DA03 - Nickname for 2nd
DA04-DA0E - Nickname for 3rd
DA0F-DA19 - Nickname for 4th
DA1A-DA24 - Nickname for 5th
DA25-DA2F - Nickname for 6th
Etc.
Game Time

DA40 - Hours, two bytes
DA42 - Minutes, two bytes
DA44 - Seconds, one byte
DA45 - Frames, one byte
DA47 - Safari Balls

Stored Pokémon
This is for the current box that's in use.

DA80 - Number of Pokémon in current box
DA81-CA94 - Pokémon 1-20
DA95 - End of list
Pokémon 1

DA96 - Pokémon
DA97-DA98 - HP
DA99 - Level
DA9A - Status
DA9B-DA9C - Types
DA9D - Held Item (When traded to Generation II)
DA9E-DAA1 - Moves
DAA2-DAA3 - Trainer ID
DAA4-DAA6 - Experience
DAA7-DAA8 - HP EV
DAA9-DAAA - Attack EV
DAAB-DAAC - Defense EV
DAAD-DAAE - Speed EV
DAAF-DAB0 - Special EV
DAB1 - Attack/Defense IV
DAB2 - Speed/Special IV
DAB3-DAB6 - PP Moves
Pokémon 2

DAB7 - Pokémon
DAB8-DAB9 - HP
DABA - Level
DABB - Status
DABC-DABD - Types
DABE - Held Item (When traded to Generation II)
DABF-DAC2 - Moves
DAC2-DAC4 - Trainer ID
DAC4-DAC7 - Experience
DAC8-DAC9 - HP EV
DACA-DACB - Attack EV
DACC-DACD - Defense EV
DACE-DACF - Speed EV
DAD0-DAD1 - Special EV
DAD2 - Attack/Defense IV
DAD3 - Speed/Special IV
DAD4-DAD7 - PP Moves
Pokémon 3

DAD8 - Pokémon
DAD9-DADA - HP
DADB - Level
DADC - Status
DADD-DADE - Types
DADF - Held Item (When traded to Generation II)
DAE0-DAE3 - Moves
DAE4-DAE5 - Trainer ID
DAE6-DAE8 - Experience
DAE9-DAEA - HP EV
DAEB-DAEC - Attack EV
DAED-DAEE - Defense EV
DAEF-DAF0 - Speed EV
DAF1-DAF2 - Special EV
DAF3 - Attack/Defense IV
DAF4 - Speed/Special IV
DAF5-DAF8 - PP Moves
Pokémon 4

DAF9 - Pokémon
DAFA-DAFB - HP
DAFC - Level
DAFD - Status
DAFE-DAFF - Types
DB00 - Held Item (When traded to Generation II)
DB01-DB04 - Moves
DB05-DB06 - Trainer ID
DB07-DB09 - Experience
DB0A-DB0B - HP EV
DB0C-DB0D - Attack EV
DB0E-DB0F - Defense EV
DB10-DB11 - Speed EV
DB12-DB13 - Special EV
DB14 - Attack/Defense IV
DB15 - Speed/Special IV
DB16-DB19 - PP Moves
Pokémon 5

DB1A - Pokémon
DB1B-DB1C - HP
DB1D - Level
DB1E - Status
DB1F-DB20 - Types
DB21 - Held Item (When traded to Generation II)
DB22-DB25 - Moves
DB26-DB27 - Trainer ID
DB28-DB2A - Experience
DB2B-DB2C - HP EV
DB2D-DB2E - Attack EV
DB2F-DB30 - Defense EV
DB31-DB32 - Speed EV
DB33-DB34 - Special EV
DB35 - Attack/Defense IV
DB36 - Speed/Special IV
DB37-DB3A - PP Moves
Pokémon 6

DB3B - Pokémon
DB3C-DB3D - HP
DB3E - Level
DB3F - Status
DB40-DB41 - Types
DB42 - Held Item (When traded to Generation II)
DB43-DB46 - Moves
DB47-DB48 - Trainer ID
DB4A-DB4B - Experience
DB4C-DB4D - HP EV
DB4E-DB4F - Attack EV
DB50-DB51 - Defense EV
DB52-DB53 - Speed EV
DB54-DB55 - Special EV
DB56 - Attack/Defense IV
DB57 - Speed/Special IV
DB58-DB5B - PP Moves
Pokémon 7

DB5C - Pokémon
DB5D-DB5E - HP
DB5F - Level
DB60 - Status
DB61-DB62 - Types
DB63 - Held Item (When traded to Generation II)
DB64-DB67 - Moves
DB68-DB69 - Trainer ID
DB6A-DB6C - Experience
DB6D-DB6E - HP EV
DB6F-DB70 - Attack EV
DB71-DB72 - Defense EV
DB73-DB74 - Speed EV
DB75-DB76 - Special EV
DB77 - Attack/Defense IV
DB78 - Speed/Special IV
DB79-DB7C - PP Moves
Pokémon 8

DB7D - Pokémon
DB7E-DB7F - HP
DB80 - Level
DB81 - Status
DB82-DB83 - Types
DB84 - Held Item (When traded to Generation II)
DB85-DB88 - Moves
DB89-DB8A - Trainer ID
DB8B-DB8D - Experience
DB8E-DB8F - HP EV
DB90-DB91 - Attack EV
DB92-DB93 - Defense EV
DB94-DB95 - Speed EV
DB96-DB97 - Special EV
DB98 - Attack/Defense IV
DB99 - Speed/Special IV
DB9A-DB9D - PP Moves
Pokémon 9

DB9E - Pokémon
DB9F-DBA0 - HP
DBA1 - Level
DBA2 - Status
DBA3-DBA4 - Types
DBA5 - Held Item (When traded to Generation II)
DBA6-DBA9 - Moves
DBAA-DBAB - Trainer ID
DBAC-DBAE - Experience
DBAF-DBB0 - HP EV
DBB1-DBB2 - Attack EV
DBB3-DBB4 - Defense EV
DBB5-DBB6 - Speed EV
DBB7-DBB8 - Special EV
DBB9 - Attack/Defense IV
DBBA - Speed/Special IV
DBBB-DBBE - PP Moves
Pokémon 10

DBBF - Pokémon
DBC0-DBC1 - HP
DBC2 - Level
DBC3 - Status
DBC4-DBC5 - Types
DBC6 - Held Item (When traded to Generation II)
DBC7-DBCA - Moves
DBCB-DBCC - Trainer ID
DBCD-DBCF - Experience
DBD0-DBD1 - HP EV
DBD2-DBD3 - Attack EV
DBD4-DBD5 - Defense EV
DBD6-DBD7 - Speed EV
DBD8-DBD9 - Special EV
DBDA - Attack/Defense IV
DBDB - Speed/Special IV
DBDC-DBDF - PP Moves
Pokémon 11

DBE0 - Pokémon
DBE1-DBE2 - HP
DBE3 - Level
DBE4 - Status
DBE5-DBE6 - Types
DBE7 - Held Item (When traded to Generation II)
DBE8-DBEB - Moves
DBEC-DBED - Trainer ID
DBEE-DBF0 - Experience
DBF1-DBF2 - HP EV
DBF3-DBF4 - Attack EV
DBF5-DBF6 - Defense EV
DBF7-DBF8 - Speed EV
DBF9-DBFA - Special EV
DBFB - Attack/Defense IV
DBFC - Speed/Special IV
DBFD-DC00 - PP Moves
Pokémon 12

DC01 - Pokémon
DC02-DC03 - HP
DC04 - Level
DC05 - Status
DC06-DC07 - Types
DC08 - Held Item (When traded to Generation II)
DC09-DC0C - Moves
DC0D-DC0E - Trainer ID
DC0F-DC11 - Experience
DC12-DC13 - HP EV
DC14-DC15 - Attack EV
DC16-DC17 - Defense EV
DC18-DC19 - Speed EV
DC1A-DC1B - Special EV
DC1C - Attack/Defense IV
DC1D - Speed/Special IV
DC1E-DC21 - PP Moves
Pokémon 13

DC22 - Pokémon
DC23-DC24 - HP
DC25 - Level
DC26 - Status
DC27-DC28 - Types
DC29 - Held Item (When traded to Generation II)
DC2A-DC2D - Moves
DC2E-DC2F - Trainer ID
DC30-DC32 - Experience
DC33-DC34 - HP EV
DC35-DC36 - Attack EV
DC37-DC38 - Defense EV
DC39-DC3A - Speed EV
DC3B-DC3C - Special EV
DC3D - Attack/Defense IV
DC3E - Speed/Special IV
DC3F-DC42 - PP Moves
Pokémon 14

DC43 - Pokémon
DC44-DC45 - HP
DC46 - Level
DC47 - Status
DC48-DC49 - Types
DC4A - Held Item (When traded to Generation II)
DC4B-DC4E - Moves
DC4F-DC50 - Trainer ID
DC51-DC53 - Experience
DC54-DC55 - HP EV
DC56-DC57 - Attack EV
DC58-DC59 - Defense EV
DC5A-DC5B - Speed EV
DC5C-DC5D - Special EV
DC5E - Attack/Defense IV
DC5F - Speed/Special IV
DC60-DC63 - PP Moves
Pokémon 15

DC64 - Pokémon
DC65-CC66 - HP
DC67 - Level
DC68 - Status
DC69-DC6A - Types
DC6B - Held Item (When traded to Generation II)
DC6C-DC6F - Moves
DC70-DC71 - Trainer ID
DC72-DC74 - Experience
DC75-DC76 - HP EV
DC77-DC78 - Attack EV
DC79-DC7A - Defense EV
DC7B-DC7C - Speed EV
DC7D-DC7E - Special EV
DC7F - Attack/Defense IV
DC80 - Speed/Special IV
DC81-DC84 - PP Moves
Pokémon 16

DC85 - Pokémon
DC86-DC87 - HP
DC88 - Level
DC89 - Status
DC8A-DC8B - Types
DC8C - Held Item (When traded to Generation II)
DC8D-DC90 - Moves
DC91-DC92 - Trainer ID
DC93-DC95 - Experience
DC96-DC97 - HP EV
DC98-DC99 - Attack EV
DC9A-DC9B - Defense EV
DC9C-DC9D - Speed EV
DC9E-DC9F - Special EV
DCA0 - Attack/Defense IV
DCA1 - Speed/Special IV
DCA2-DCA5 - PP Moves
Pokémon 17

DCA6 - Pokémon
DCA7-DCA8 - HP
DCA9 - Level
DCAA - Status
DCAB-DCAC - Types
DCAD - Held Item (When traded to Generation II)
DCAE-DCB1 - Moves
DCB2-DCB3 - Trainer ID
DCB4-DCB6 - Experience
DCB7-DCB8 - HP EV
DCB9-DCBA - Attack EV
DCBB-DCBC - Defense EV
DCBD-DCBE - Speed EV
DCBF-DCC0 - Special EV
DCC1 - Attack/Defense IV
DCC2 - Speed/Special IV
DCC3-DCC6 - PP Moves
Pokémon 18

DCC7 - Pokémon
DCC8-DCC9 - HP
DCCA - Level
DCCB - Status
DCCC-DCCD - Types
DCCE - Held Item (When traded to Generation II)
DCCF-DCD2 - Moves
DCD3-DCD4 - Trainer ID
DCD5-DCD7 - Experience
DCD8-DCD9 - HP EV
DCDA-DCDB - Attack EV
DCDC-DCDD - Defense EV
DCDE-DCDF - Speed EV
DCE0-DCE1 - Special EV
DCE2 - Attack/Defense IV
DCE3 - Speed/Special IV
DCE4-DCE7 - PP Moves
Pokémon 19

DCE8 - Pokémon
DCE9-DCEA - HP
DCEB - Level
DCEC - Status
DCED-DCEE - Types
DCEF - Held Item (When traded to Generation II)
DCF0-DCF3 - Moves
DCF4-DCF5 - Trainer ID
DCF6-DCF8 - Experience
DCF9-DCFA - HP EV
DCFB-DCFC - Attack EV
DCFD-DCFE - Defense EV
DCFF-DD00 - Speed EV
DD01-DD02 - Special EV
DD03 - Attack/Defense IV
DD04 - Speed/Special IV
DD05-DD08 - PP Moves
Pokémon 20

DD09 - Pokémon
DD0A-DD0B - HP
DD0C - Level
DD0D - Status
DD0E-DD0F - Types
DD10 - Held Item (When traded to Generation II)
DD11-DD14 - Moves
DD15-DD16 - Trainer ID
DD17-DD19 - Experience
DD1A-DD1B - HP EV
DD1C-DD1D - Attack EV
DD1E-DD1F - Defense EV
DD20-DD21 - Speed EV
DD22-DD23 - Special EV
DD24 - Attack/Defense IV
DD25 - Speed/Special IV
DD26-DD29 - PP Moves
Trainer name

DD2A-DD34 - Trainer name for 1st
DD35-DD3F - Trainer name for 2nd
DD40-DD4A - Trainer name for 3rd
DD4B-DD55 - Trainer name for 4th
DD56-DD60 - Trainer name for 5th
DD61-DD6B - Trainer name for 6th
DD6C-DD76 - Trainer name for 7th
DD77-DD81 - Trainer name for 8th
DD82-DD8C - Trainer name for 9th
DD8D-DD97 - Trainer name for 10th
DD98-DDA2 - Trainer name for 11th
DDA3-DDAD - Trainer name for 12th
DDAE-DDB8 - Trainer name for 13th
DDB9-DDC3 - Trainer name for 14th
DDC4-DDCE - Trainer name for 15th
DDCF-DDD9 - Trainer name for 16th
DDDA-DDE4 - Trainer name for 17th
DDE5-DDEF - Trainer name for 18th
DDF0-DDFA - Trainer name for 19th
DDFB-DE05 - Trainer name for 20th
Nickname

DE06-DE10 - Nickname for 1st
DE11-DE1B - Nickname for 2nd
DE1C-DE26 - Nickname for 3rd
DE27-DE31 - Nickname for 4th
DE32-DE3C - Nickname for 5th
DE3D-DE47 - Nickname for 6th
DE48-DE52 - Nickname for 7th
DE53-DE5D - Nickname for 8th
DE5E-DE68 - Nickname for 9th
DE69-DE73 - Nickname for 10th
DE74-DE7E - Nickname for 11th
DE7F-DED9 - Nickname for 12th
DE8A-DE94 - Nickname for 13th
DE95-DE9F - Nickname for 14th
DEA0-DEAA - Nickname for 15th
DEAB-DEB5 - Nickname for 16th
DEB6-DEC0 - Nickname for 17th
DEC1-DECB - Nickname for 18th
DECC-DED6 - Nickname for 19th
DED7-DEE1 - Nickname for 20th


Internal data (HRAM)
When the game boots, HRAM (FF80 to FFFE) is initialized to 0.

OAM DMA
FF80-FF89 - OAM DMA routine
Misc.
FF8A - Soft reset frame counter
Initially has value 16.
Decremented each frame START+SELECT+A+B are held down ; when 0, the game soft resets.
 FF9F-FFA1 - Money / Coins amount (big-endian)
When holding coins amount, 0 is written to FF9F for some reason.
Serial
FFA9-FFAD - Serial status data (sync method, etc.)
Graphics scroll values
FFAE - Value copied to SCX at VBlank
FFAF - Value copied to SCY at VBlank
FFB0 - Value copied to WY  at VBlank
Joypad
FFB1 - Joypad input during previous frame
FFB2 - Released buttons on this frame
FFB3 - Pressed buttons on this frame
FFB4 - Held buttons on this frame
 FFB5 - Low-sensitivity joypad output, controlled by FFB6 and FFB7.
FFB6 - If zero and A or B is held, outputs nothing. Otherwise, considers buttons held for more than half a second being pressed twelve times a second.
Has no influence if FFB7 is zero.
FFB7 - If zero, FFB5 contains newly-pressed buttons only.
ROM banking
FFB8 - Loaded ROM bank.
FFB9 - Saved ROM bank.
VBlank data transfer
FFBA - Transfer background to VRAM during VBlank ?
FFBB - Which portion of the screen to transfer ? (0-1-2 = top-middle-bottom)
 FFBC-FFBD - Where to copy background to ? (does not account for FFBB offset ; low-endian address)
 FFBF-FFC0 - Stack pointer save
 FFC1-FFC2 - Where to copy background from ? (used during copy ; low-endian address)
FFBA is copied to FFC1 (LSB of address).
FFC3-FFC4 - Where to copy background to ? (low-endian address)
FFC5 - Number of rows to copy
 FFC6 - Number of 16-byte units to copy during VBlank
FFC7-FFC8 - Source
FFC9-FFCA - Destination
 FFCB - Number of 8-byte units to copy during VBlank
FFCC-FFCD - Source
FFCE-FFCF - Destination
VBlank 2x2 tile block redraw
FFD0 - What to redraw (0-1-2 = nothing-column-row)
FFD1-FFD2 - Destination
RNG
FFD3 - RandomAdd
FFD4 - RandomSub
On each call to the Random function, the value of register rDIV (roughly tracking how many bytes were executed) and the carry are added to RandomAdd, and subtracted from RandomSub.

VBlank interface
FFD5 - Frame counter (decremented each VBlank)
FFD6 - Set to 0 when a VBlank occurs
Misc. 2
FFD7 - Tileset type (0-1-2 = indoors-cave-outside)
Nonzero values have the water animation.
Value 2 has the flower animation.
 FFF3 - Battle turn (0-1 = player-opponent)
 FFF8 - Joypad input
FFF9 - Diable joypad polling
Internal Data for Pokémon Red and Blue
ROM map • RAM map • Text table • Notes • Tutorials
Categories: RAM mapsTo do
Navigation menu
Create account
Log in
PageDiscussion
ReadView sourceView history
Search
Search Data Crystal
Main Page
Recent changes
Random page
Help
Community
ROMhack.ing
TCRF
Discord (RHDI)
Discord (TCRF)
Fan Translation Community History
Games
By platform
By series
By content
By hardware
By release date
By developer
By publisher
RHDI Games List
Translation Request List
Tools
What links here
Related changes
Special pages
Printable version
Permanent link
Page information
This page was last edited on 4 May 2025, at 05:17.
Content is available under GNU Free Documentation License 1.2 unless otherwise noted.
Privacy policyAbout Data CrystalDisclaimers
GNU Free Documentation License 1.2Powered by MediaWiki