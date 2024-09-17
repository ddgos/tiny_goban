#![cfg_attr(not(test), no_std)]

use core::ops::Range;

use heapless::{spsc::Queue, String, Vec};

#[derive(Debug, PartialEq, Clone)]
pub enum Player {
    Black,
    White,
}

#[derive(Debug, PartialEq)]
pub enum KoState {
    CapturedInKo,
    Otherwise,
}

#[derive(Debug, PartialEq)]
pub enum Point {
    Stone(Player),
    // ko occurs when a played stone has 1 liberty remaining after being played
    //   and a single stone was captured to get to this position
    // In this situation, playing in the captured location is forbidden
    Clear(KoState),
}

impl Point {
    const BLACK_POINT_CODE: u8 = 0b11000000;
    const WHITE_POINT_CODE: u8 = 0b10000000;
    const CLEAR_POINT_CODE: u8 = 0b00000000;
    const CAPTURED_IN_KO_POINT_CODE: u8 = 0b01000000;

    const fn from_valid_code_point(code_point: u8) -> Point {
        match code_point {
            Point::BLACK_POINT_CODE => Point::Stone(Player::Black),
            Point::WHITE_POINT_CODE => Point::Stone(Player::White),
            Point::CLEAR_POINT_CODE => Point::Clear(KoState::Otherwise),
            Point::CAPTURED_IN_KO_POINT_CODE => Point::Clear(KoState::CapturedInKo),
            _ => panic!("unexpected code point"),
        }
    }

    const fn to_code_point(&self) -> u8 {
        match self {
            Self::Stone(Player::Black) => Self::BLACK_POINT_CODE,
            Self::Stone(Player::White) => Self::WHITE_POINT_CODE,
            Self::Clear(KoState::Otherwise) => Self::CLEAR_POINT_CODE,
            Self::Clear(KoState::CapturedInKo) => Self::CAPTURED_IN_KO_POINT_CODE,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Coord {
    index: u16,
}

impl Coord {
    pub const fn one_index_new(x: u8, y: u8) -> Option<Coord> {
        Self::new(x - 1, y - 1)
    }

    pub const fn new(x: u8, y: u8) -> Option<Coord> {
        if x < 19 && y < 19 {
            Some(Coord::new_unchecked(x, y))
        } else {
            None
        }
    }

    const fn new_unchecked(x: u8, y: u8) -> Coord {
        Coord {
            index: (y as u16 * 19) + x as u16,
        }
    }

    pub const fn x(&self) -> u8 {
        self.index.rem_euclid(19) as u8
    }

    pub const fn y(&self) -> u8 {
        self.index.div_euclid(19) as u8
    }

    pub const fn to_xy(&self) -> (u8, u8) {
        (self.x(), self.y())
    }

    pub const fn to_one_index_xy(&self) -> (u8, u8) {
        (self.x() + 1, self.y() + 1)
    }

    fn neighbour_in_direction(&self, direction: &Direction) -> Option<Coord> {
        match direction {
            Direction::North => self.northern_neighbour(),
            Direction::East => self.eastern_neighbour(),
            Direction::South => self.southern_neighbour(),
            Direction::West => self.western_neighbour(),
        }
    }

    fn southern_neighbour(&self) -> Option<Coord> {
        let (x, y) = self.to_xy();
        y.checked_sub(1)
            .map(|smaller_y| Coord::new(x, smaller_y).expect("should be within ranges"))
    }

    fn eastern_neighbour(&self) -> Option<Coord> {
        let (x, y) = self.to_xy();
        Coord::new(x + 1, y)
    }

    fn northern_neighbour(&self) -> Option<Coord> {
        let (x, y) = self.to_xy();
        Coord::new(x, y + 1)
    }

    fn western_neighbour(&self) -> Option<Coord> {
        let (x, y) = self.to_xy();
        x.checked_sub(1)
            .map(|smaller_x| Coord::new(smaller_x, y).expect("should be within ranges"))
    }

    fn neighbours(&self) -> Vec<Coord, 4> {
        Vec::<Coord, 4>::from_iter(
            Direction::NSEW
                .iter()
                .filter_map(|direction| self.neighbour_in_direction(direction)),
        )
    }
}

/*
    empty: 00
    empty: 01 - liberty of single stone that just captured single stone
    black: 11
    white: 10
     1] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8
     2] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- --
     3] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <-
     4] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 ->
     5] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8
     6] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- --
     7] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <-
     8] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 ->
     9] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8
    10] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- --
    11] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <-
    12] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 ->
    13] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8
    14] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- --
    15] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <-
    16] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 ->
    17] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8
    18] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- --
    19] | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
         u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 -> <- -- u8 ->
*/

enum CodePointIndex {
    First,
    Second,
    Third,
    Fourth,
}

impl CodePointIndex {
    const ALL: [Self; 4] = [Self::First, Self::Second, Self::Third, Self::Fourth];
}

#[derive(Default, Clone, Debug, PartialEq)]
struct FourPointStore(u8);

impl FourPointStore {
    const CODE_POINT_MASK: u8 = 0b11000000;

    const fn points(&self) -> [Point; 4] {
        [
            Point::from_valid_code_point(self.0.rotate_left(0) & FourPointStore::CODE_POINT_MASK),
            Point::from_valid_code_point(self.0.rotate_left(2) & FourPointStore::CODE_POINT_MASK),
            Point::from_valid_code_point(self.0.rotate_left(4) & FourPointStore::CODE_POINT_MASK),
            Point::from_valid_code_point(self.0.rotate_left(6) & FourPointStore::CODE_POINT_MASK),
        ]
    }

    const fn get(&self, index: &CodePointIndex) -> Point {
        let [point_1, point_2, point_3, point_4] = self.points();
        match index {
            CodePointIndex::First => point_1,
            CodePointIndex::Second => point_2,
            CodePointIndex::Third => point_3,
            CodePointIndex::Fourth => point_4,
        }
    }

    fn set(&mut self, code_index: &CodePointIndex, point: Point) {
        let (right_shifts, new_location_mask) = match code_index {
            CodePointIndex::First => (0, 0b11000000),
            CodePointIndex::Second => (2, 0b00110000),
            CodePointIndex::Third => (4, 0b00001100),
            CodePointIndex::Fourth => (6, 0b00000011),
        };
        // clear the bits
        self.0 &= !new_location_mask;
        // set the bits
        self.0 |= point.to_code_point().rotate_right(right_shifts)
    }

    fn clear_ko_store(&mut self) {
        // optimisations:
        //  - there should only be one of these, so stop after one is found
        //  - fiddle with the bits!
        for index in CodePointIndex::ALL {
            if self.get(&index) == Point::Clear(KoState::CapturedInKo) {
                self.set(&index, Point::Clear(KoState::Otherwise))
            }
        }
    }
}

#[derive(Default, Clone, Debug, PartialEq)]
struct FinalPointAndKoStore(u8);

macro_rules! range_for {
    ($min:expr, $max:expr, $head:expr) => {{
        let new_boundary = ($min + $max) / 2;

        match $head {
            true => (new_boundary..$max),
            false => ($min..new_boundary),
        }
    }};

    ($min:expr, $max:expr, $head:expr, $($tail:expr),+) => {{
        let new_boundary = ($min + $max) / 2;

        let (new_min, new_max) = match $head {
            true => (new_boundary, $max),
            false => ($min, new_boundary),
        };

        range_for!(new_min, new_max, $($tail),+)
    }};
}

impl FinalPointAndKoStore {
    const POINT_MASK: u8 = 0b11000000;
    const KO_EXISTS_MASK: u8 = 0b00100000;

    const fn point(&self) -> Point {
        Point::from_valid_code_point(self.0 & FinalPointAndKoStore::POINT_MASK)
    }

    // this could be calculated at compile time and run as a lookup function
    const fn ko_store_range(&self) -> Option<Range<u8>> {
        if self.0 & FinalPointAndKoStore::KO_EXISTS_MASK == 0 {
            return None;
        }

        let c1 = self.0 & 0b00010000 == 0b00010000;
        let c2 = self.0 & 0b00001000 == 0b00001000;
        let c3 = self.0 & 0b00000100 == 0b00000100;
        let c4 = self.0 & 0b00000010 == 0b00000010;
        let c5 = self.0 & 0b00000001 == 0b00000001;

        Some(range_for!(0, 92, c1, c2, c3, c4, c5))
    }

    fn set_ko_store(&mut self, ko_capture_coord: &Coord) {
        let final_point = self.0 & FinalPointAndKoStore::POINT_MASK;
        // optimisation: calculate these at compile time and match
        // all ko stores possible
        let ko_store_value = (0b00100000..=0b00111111u8)
            // find the one that includes this coord
            .find(|store_code| {
                let store = Self(*store_code);
                store
                    .could_have_ko_at(&ko_capture_coord)
                    .expect("should be ko-style store")
            })
            .expect("one of the store should accept this coord");

        // put the code back together
        self.0 = final_point | ko_store_value;
    }

    fn could_have_ko_at(&self, coord: &Coord) -> Option<bool> {
        let store_index = Goban::coord_to_store_index(coord);
        self.ko_store_range()
            .map(|range| range.contains(&(store_index as u8)))
    }

    fn set_point(&mut self, point: Point) {
        self.0 &= !Self::POINT_MASK;
        self.0 |= point.to_code_point()
    }

    fn clear_ko_store(&mut self) {
        if self.point() == Point::Clear(KoState::CapturedInKo) {
            self.set_point(Point::Clear(KoState::Otherwise))
        }
        self.0 &= Self::POINT_MASK;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Goban {
    main_points_store: [FourPointStore; 90],
    final_point_and_ko_store: FinalPointAndKoStore,
}

impl Default for Goban {
    fn default() -> Self {
        Self {
            main_points_store: [(); 90].map(|_| FourPointStore::default()),
            final_point_and_ko_store: FinalPointAndKoStore::default(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum GobanPlayError {
    OccupiedIntersection,
    KoRuleViolation,
}

type PlayKoResult = Option<Coord>;

#[derive(Debug)]
enum Direction {
    North,
    East,
    South,
    West,
}

impl Direction {
    const NSEW: [Direction; 4] = [
        Direction::North,
        Direction::East,
        Direction::South,
        Direction::West,
    ];
}

enum SingleOccuranceTracker<T> {
    None,
    One(T),
    TooMany,
}

impl<T> SingleOccuranceTracker<T> {
    fn update_with(&self, value: T) -> Self {
        match self {
            Self::None => Self::One(value),
            _ => Self::TooMany,
        }
    }
}

type KoCaptureTracker = SingleOccuranceTracker<Direction>;

impl Goban {
    pub fn new() -> Goban {
        Default::default()
    }

    pub fn pretty_print(&self) -> String<380> {
        let mut s = String::new();
        for y in (0..19).rev() {
            for x in 0..19 {
                let this_coord = Coord::new(x, y).expect("x and y should be within valid range");
                let this_point = self.get(&this_coord);
                s.push(match this_point {
                    Point::Stone(Player::Black) => 'x',
                    Point::Stone(Player::White) => 'o',
                    Point::Clear(KoState::Otherwise) => {
                        if [3, 9, 15].contains(&x) && [3, 9, 15].contains(&y) {
                            '-'
                        } else {
                            '.'
                        }
                    }
                    Point::Clear(KoState::CapturedInKo) => '~',
                })
                .expect("should have capacity for entire string")
            }
            s.push('\n')
                .expect("should have capacity for final character")
        }
        s
    }

    // ko occurs when a played stone has 1 liberty remaining after being played
    //   and a single stone was captured to get to this position
    // In this situation, playing in the captured location is forbidden
    // Goban is not mutated until GobanPlayErrors have been checked for,
    //   so the goban after an Error should always be equal to the goban before
    pub fn play(&mut self, coord: &Coord, player: Player) -> Result<(), GobanPlayError> {
        match self.get(coord) {
            Point::Stone(_) => return Err(GobanPlayError::OccupiedIntersection),
            Point::Clear(KoState::CapturedInKo) => return Err(GobanPlayError::KoRuleViolation),
            Point::Clear(KoState::Otherwise) => (),
        }

        let ko_result = self.raw_play(coord, player);

        self.clear_ko_store();
        if let Some(coord_captured_in_ko) = ko_result {
            self.set_ko_store(coord_captured_in_ko);
        }

        Ok(())
    }

    pub fn set(&mut self, coord: &Coord, point: Point) {
        let (point_store_index, code_index) = Goban::coord_to_store_indecies(coord);
        match point_store_index {
            90 => self.final_point_and_ko_store.set_point(point),
            main_store_index => self.main_points_store[main_store_index].set(&code_index, point),
        }
    }

    fn raw_play(&mut self, coord: &Coord, player: Player) -> PlayKoResult {
        self.set(&coord, Point::Stone(player.clone()));

        let mut ko_capture_tracker = KoCaptureTracker::None;

        // collect first as we're about to mutate self
        let was_neighbouring_opponent_coords: Vec<(Direction, Coord), 4> = Direction::NSEW
            .into_iter()
            .filter_map(|direction| {
                coord
                    .neighbour_in_direction(&direction)
                    .map(|neighbour| (direction, neighbour))
            })
            .filter(|(_direction, neighbour)| match self.get(neighbour) {
                Point::Stone(neighbour_owner) => player != neighbour_owner,
                _otherwise => false,
            })
            .collect();

        // capture opponent groups
        for (direction, where_opponent_was) in was_neighbouring_opponent_coords {
            let coords_to_clear = self.coords_to_capture(where_opponent_was);
            if coords_to_clear.len() == 1 {
                ko_capture_tracker = ko_capture_tracker.update_with(direction);
            }
            for coord in coords_to_clear {
                // this later gets corrected if there was a ko capture, so
                // it's okay if we set the ko state incorrectly for now
                self.set(&coord, Point::Clear(KoState::Otherwise))
            }
        }

        let ko_result = if self.could_have_captured_ko(coord) {
            match ko_capture_tracker {
                KoCaptureTracker::None | KoCaptureTracker::TooMany => None,
                KoCaptureTracker::One(direction) => Some(
                    coord
                        .neighbour_in_direction(&direction)
                        .expect("should be a neighbour if there was a capture"),
                ),
            }
        } else {
            None
        };

        // if let Some(ref ko_capture_coord) = ko_result {
        //     eprintln!("captured coord {:?}", ko_capture_coord.to_one_index_xy())
        // } else {
        //     eprintln!("no stones captured in ko")
        // }

        ko_result
    }

    fn clear_ko_store(&mut self) {
        if let Some(ko_range) = self.final_point_and_ko_store.ko_store_range() {
            for store in ko_range {
                match store {
                    91 => self.final_point_and_ko_store.clear_ko_store(),
                    main_sequence_store => {
                        self.main_points_store[main_sequence_store as usize].clear_ko_store()
                    }
                }
            }
        }
    }

    fn set_ko_store(&mut self, coord_captured_in_ko: Coord) {
        // sets store index 91
        self.final_point_and_ko_store
            .set_ko_store(&coord_captured_in_ko);
        let (store_index, code_point_index) = Self::coord_to_store_indecies(&coord_captured_in_ko);
        if store_index == 91 {
            // would have already been set above
        } else {
            self.main_points_store[store_index as usize]
                .set(&code_point_index, Point::Clear(KoState::CapturedInKo))
        }
    }

    const fn coord_to_store_index(coord: &Coord) -> usize {
        coord.index.div_euclid(4) as usize
    }

    const fn coord_to_store_indecies(coord: &Coord) -> (usize, CodePointIndex) {
        let code_point_index = match coord.index.rem_euclid(4) {
            0 => CodePointIndex::First,
            1 => CodePointIndex::Second,
            2 => CodePointIndex::Third,
            3 => CodePointIndex::Fourth,
            _ => panic!("result of a % b must be less than b"),
        };
        (Self::coord_to_store_index(coord), code_point_index)
    }

    pub const fn get(&self, coord: &Coord) -> Point {
        let (point_store_index, code_index) = Goban::coord_to_store_indecies(coord);
        match point_store_index {
            90 => self.final_point_and_ko_store.point(),
            main_store_index => self.main_points_store[main_store_index as usize].get(&code_index),
        }
    }

    pub fn points(&self) -> [Point; 361] {
        Vec::<Point, 361>::from_iter(self.points_iter())
            .into_array()
            .expect("should be correct length")
    }

    pub fn points_iter(&self) -> impl Iterator<Item = Point> + '_ {
        let main_points = self
            .main_points_store
            .iter()
            .flat_map(|point_store| point_store.points().into_iter());
        let final_point = [self.final_point_and_ko_store.point()].into_iter();
        main_points.chain(final_point)
    }

    fn could_have_captured_ko(&self, coord: &Coord) -> bool {
        let played_colour = match self.get(&coord) {
            Point::Stone(colour) => colour,
            Point::Clear(_) => return false,
        };

        let mut found_clear = false;

        for neighbour in coord.neighbours() {
            match self.get(&neighbour) {
                Point::Stone(neighbouring_player) => {
                    if neighbouring_player == played_colour {
                        // this is not a single stone
                        return false;
                    }
                }
                Point::Clear(_) => {
                    if found_clear {
                        // there is more than one liberty
                        return false;
                    } else {
                        // so far there is one liberty
                        found_clear = true;
                    }
                }
            }
        }

        // can only exit the loop if the stone is alone
        // and if it has 0 or 1 liberties
        // not sure what to do about 0 liberties...
        // one of these is better than the other:
        // return true;
        // return found_clear;
        found_clear
    }

    fn coords_to_capture(&self, root_coord: Coord) -> Vec<Coord, 361> {
        let clearing_colour = match self.get(&root_coord) {
            Point::Stone(colour) => colour,
            // there is no group here, so there are no coords to capture
            Point::Clear(_) => return Vec::new(),
        };
        let mut to_clear = Vec::<Coord, 361>::new();
        let mut to_process = Queue::<Coord, 361>::new();
        to_process
            .enqueue(root_coord)
            .expect("should have capacity for first");

        while let Some(this_coord) = to_process.dequeue() {
            if to_clear.contains(&this_coord) {
                // already processed by another route
                // continue to next in process queue
                continue;
            }
            let this_coord_content = self.get(&this_coord);
            match this_coord_content {
                Point::Stone(found_colour) => {
                    if found_colour == clearing_colour {
                        // add neighbours to process queue
                        for neighbour in this_coord.neighbours() {
                            to_process
                                .enqueue(neighbour)
                                .expect("should have capacity for whole board");
                        }
                        // add this to return vector
                        to_clear.push(this_coord).expect("should have capacity");
                    } else {
                        // opponent stone, no-op
                    }
                }
                // liberty, so this group should not be captured
                Point::Clear(_) => return Vec::new(),
            };
        }

        to_clear
    }
}

#[cfg(test)]
mod tests {
    use std::ops::RangeBounds;

    use crate::{Coord, FinalPointAndKoStore, Goban, GobanPlayError, KoState, Player, Point};

    #[test]
    fn all_bounds_covered() {
        let first_range = FinalPointAndKoStore(0b00100000).ko_store_range().unwrap();
        println!("range: {:?}", first_range);
        let first_lower = first_range.clone().min().unwrap();
        assert_eq!(first_lower, 0, "should start at 0");
        let mut prev_upper = first_range.max().unwrap();

        for code in (0b00100001..=0b00111111).map(FinalPointAndKoStore) {
            let range = code.ko_store_range().unwrap();
            println!("range: {:?}", range);
            match range.start_bound() {
                std::ops::Bound::Included(lower) => {
                    assert_eq!(&(prev_upper + 1), lower, "code: {}", code.0)
                }
                _ => panic!("expected included lower bound"),
            };

            prev_upper = range.max().unwrap();
        }

        assert_eq!(prev_upper, 91, "should end at 91");
    }

    #[test]
    fn all_coords_ko_store_correctly() {
        for x in 0..19 {
            for y in 0..19 {
                let coord = Coord::new(x, y).expect("should be good coord");
                let mut ko_store = FinalPointAndKoStore(0);
                ko_store.set_ko_store(&coord);
                let coord = Coord::new(x, y).expect("should be good coord");
                assert!(
                    ko_store
                        .could_have_ko_at(&coord)
                        .expect("should be storing coord"),
                    "ko store should be storing ko info about {:?}, but claims otherwise",
                    coord
                )
            }
        }
    }

    #[test]
    fn can_play_simple() {
        let mut goban = Goban::default();
        let plays = [
            (Player::Black, 16, 4),
            (Player::White, 17, 16),
            (Player::Black, 4, 17),
            (Player::White, 3, 3),
            (Player::Black, 15, 17),
            (Player::White, 16, 14),
            (Player::Black, 17, 12),
            (Player::White, 17, 10),
            (Player::Black, 15, 12),
            (Player::White, 17, 7),
            (Player::Black, 17, 17),
            (Player::White, 18, 17),
            (Player::Black, 18, 18),
            (Player::White, 16, 17),
            (Player::Black, 17, 18),
            (Player::White, 15, 16),
            (Player::Black, 16, 18),
            (Player::White, 16, 16),
            (Player::Black, 18, 16),
            (Player::White, 18, 15),
            (Player::Black, 19, 17),
            (Player::White, 14, 17),
            (Player::Black, 15, 18),
            (Player::White, 13, 16),
        ]
        .map(|(player, x, y)| (player, Coord::one_index_new(x, y).unwrap()));
        for (move_num, (player, coord)) in plays.into_iter().enumerate() {
            println!("{}: {:?}, {:?}", move_num, player, coord);
            goban.play(&coord, player).unwrap();
            println!("{}", goban.pretty_print());
        }

        let expected_captured_coord = Coord::one_index_new(18, 17).unwrap();

        assert_eq!(
            goban.get(&expected_captured_coord),
            Point::Clear(KoState::Otherwise)
        )
    }

    #[test]
    fn can_get_final_point() {
        let goban = Goban::default();
        let final_point = Coord { index: 360 };
        goban.get(&final_point);
    }

    #[test]
    fn can_get_final_point_another_way() {
        let goban = Goban::default();
        let final_point = Coord::new(18, 18).unwrap();
        goban.get(&final_point);
    }

    #[test]
    fn cant_get_beyond_final_point() {
        let goban = Goban::default();
        let beyond_final_point = Coord { index: 361 };
        goban.get(&beyond_final_point);
    }

    #[test]
    fn can_get_all_points() {
        let goban = Goban::default();
        for x in 0..19 {
            for y in 0..19 {
                let this_coord = Coord::new(x, y).unwrap();
                goban.get(&this_coord);
            }
        }
    }

    #[test]
    fn can_set_all_points() {
        let mut goban = Goban::default();
        for x in 0..19 {
            for y in 0..19 {
                let this_coord = Coord::new(x, y).unwrap();
                goban.set(&this_coord, Point::Clear(KoState::Otherwise));
            }
        }
    }

    #[test]
    fn tengen_north_neighbour_is_as_expected() {
        let tengen = Coord::one_index_new(10, 10).unwrap();
        let expected_north = Coord::one_index_new(10, 11).unwrap();
        assert_eq!(tengen.northern_neighbour().unwrap(), expected_north);
    }

    #[test]
    fn tengen_east_neighbour_is_as_expected() {
        let tengen = Coord::one_index_new(10, 10).unwrap();
        let expected_east = Coord::one_index_new(11, 10).unwrap();
        assert_eq!(tengen.eastern_neighbour().unwrap(), expected_east);
    }

    #[test]
    fn tengen_south_neighbour_is_as_expected() {
        let tengen = Coord::one_index_new(10, 10).unwrap();
        let expected_south = Coord::one_index_new(10, 9).unwrap();
        assert_eq!(tengen.southern_neighbour().unwrap(), expected_south);
    }

    #[test]
    fn tengen_west_neighbour_is_as_expected() {
        let tengen = Coord::one_index_new(10, 10).unwrap();
        let expected_west = Coord::one_index_new(9, 10).unwrap();
        assert_eq!(tengen.western_neighbour().unwrap(), expected_west);
    }

    #[test]
    fn capture_works() {
        let captured_coord = Coord::one_index_new(1, 1).unwrap();
        let capture_setup_coord = Coord::one_index_new(2, 1).unwrap();
        let capture_coord = Coord::one_index_new(1, 2).unwrap();

        assert_eq!(
            captured_coord.eastern_neighbour(),
            Some(capture_setup_coord.clone())
        );
        assert_eq!(
            captured_coord.northern_neighbour(),
            Some(capture_coord.clone())
        );
        assert_eq!(captured_coord.southern_neighbour(), None);
        assert_eq!(captured_coord.western_neighbour(), None);

        let mut goban = Goban::default();
        goban.play(&captured_coord, Player::White).unwrap();
        goban.play(&capture_setup_coord, Player::Black).unwrap();

        assert_eq!(goban.get(&captured_coord), Point::Stone(Player::White));

        goban.play(&capture_coord, Player::Black).unwrap();

        assert_eq!(goban.get(&captured_coord), Point::Clear(KoState::Otherwise));
    }

    #[test]
    fn xys_correct() {
        for x in 0..19 {
            for y in 0..19 {
                let this_coord = Coord::new(x, y).unwrap();
                assert_eq!(x, this_coord.x());
                assert_eq!(y, this_coord.y());
            }
        }
    }

    #[test]
    fn simple_ko_forbidden() {
        let mut goban = Goban::default();
        let white_ponnuki_centre = Coord::one_index_new(3, 3).unwrap();
        let black_ponnuki_centre = Coord::one_index_new(2, 3).unwrap();
        // White has a ponnuki, with center at (3, 3)
        [(3, 2), (2, 3), (4, 3), (3, 4)]
            .map(|(x, y)| Coord::one_index_new(x, y).unwrap())
            .into_iter()
            .for_each(|coord| goban.play(&coord, Player::White).unwrap());
        // Black has the White (2, 3) stone in atari
        [(2, 2), (1, 3), (2, 4)]
            .map(|(x, y)| Coord::one_index_new(x, y).unwrap())
            .into_iter()
            .for_each(|coord| goban.play(&coord, Player::Black).unwrap());

        eprintln!("{}", goban.pretty_print());
        goban.play(&white_ponnuki_centre, Player::Black).unwrap();
        eprintln!("{}", goban.pretty_print());

        assert_eq!(
            goban.get(&black_ponnuki_centre),
            Point::Clear(KoState::CapturedInKo),
            "stone at {:?} should have been captured in ko",
            black_ponnuki_centre.to_one_index_xy()
        );

        let err = goban
            .play(&black_ponnuki_centre, Player::White)
            .expect_err("should have played where Black just took the ko");

        assert_eq!(err, GobanPlayError::KoRuleViolation);
    }

    #[test]
    fn recapture_allowed_after_ko_threats() {
        let mut goban = Goban::default();
        let white_ponnuki_centre = Coord::one_index_new(3, 3).unwrap();
        let black_ponnuki_centre = Coord::one_index_new(2, 3).unwrap();
        // White has a ponnuki, with center at (3, 3)
        [(3, 2), (2, 3), (4, 3), (3, 4)]
            .map(|(x, y)| Coord::one_index_new(x, y).unwrap())
            .into_iter()
            .for_each(|coord| goban.play(&coord, Player::White).unwrap());
        // Black has the White (2, 3) stone in atari
        [(2, 2), (1, 3), (2, 4)]
            .map(|(x, y)| Coord::one_index_new(x, y).unwrap())
            .into_iter()
            .for_each(|coord| goban.play(&coord, Player::Black).unwrap());

        eprintln!("{}", goban.pretty_print());
        goban.play(&white_ponnuki_centre, Player::Black).unwrap();
        eprintln!("{}", goban.pretty_print());

        assert_eq!(
            goban.get(&black_ponnuki_centre),
            Point::Clear(KoState::CapturedInKo),
            "stone at {:?} should have been captured in ko",
            black_ponnuki_centre.to_one_index_xy()
        );

        let goban_before = goban.clone();

        let err = goban
            .play(&black_ponnuki_centre, Player::White)
            .expect_err("should have played where Black just took the ko");

        assert_eq!(err, GobanPlayError::KoRuleViolation);

        assert_eq!(
            goban_before, goban,
            "goban should be unchanged after bad play"
        );

        eprintln!("{}", goban.pretty_print());

        // very threatening tengen move...
        goban
            .play(&Coord::one_index_new(10, 10).unwrap(), Player::White)
            .unwrap();
        eprintln!("{}", goban.pretty_print());
        goban
            .play(&Coord::one_index_new(11, 10).unwrap(), Player::Black)
            .unwrap();
        eprintln!("{}", goban.pretty_print());

        goban
            .play(&black_ponnuki_centre, Player::White)
            .expect("playing the ko threats makes this move allowed now");
        eprintln!("{}", goban.pretty_print());
    }
}
