from PIL import Image

class SemanticNetVisual:
    def __init__(self, problem):
        self.problem = problem

        self.row1 = {}
        self.row2 = {}
        self.row3 = {}
        self.objects = {}
        self.figures = []
        self.solutions = []
        self.answers = {}

        self.row1_labels = []
        self.row2_labels = []
        self.row3_labels = []

        self.col1_labels = []
        self.col2_labels = []
        self.col3_labels = []

        self.horizontal_increment_probability = 1.0
        self.vertical_increment_probability = 1.0

        self.horizontal_addition_probability = 1.0
        self.vertical_addition_probability = 1.0

        self.horizontal_deletion_probability = 1.0
        self.vertical_deletion_probability = 1.0

        self.horizontal_no_change_probability = 1.0
        self.vertical_no_change_probability = 1.0

        self.position_change_probability = [1, 1, 1, 1, 1]

        self.differing_horiz_objects_probability = 1.0
        self.differing_vert_objects_probability = 1.0

        self.all_probabilities = [self.horizontal_increment_probability, self.vertical_increment_probability,\
        self.horizontal_addition_probability, self.vertical_addition_probability,\
        self.horizontal_deletion_probability,self.vertical_deletion_probability,\
        self.horizontal_no_change_probability, self.vertical_no_change_probability,\
        self.differing_horiz_objects_probability,self.differing_vert_objects_probability]
        self.solution_labels = []

        # self.file_dir = ""
        self.images = {}
        self.black_pixels_frame = {}
        self.black_pixels_solutions = {}

        self.solution_probabilities = []
        # self.

        if problem.problemType == "3x3":
            self.row1_labels = ["A", "B", "C"]
            self.row2_labels = ["D", "E", "F"]
            self.row3_labels = ["G", "H"]

            self.col1_labels = ["A", "D", "G"]
            self.col2_labels = ["B", "E", "H"]
            self.col3_labels = ["C", "F"]
            self.solution_labels = ["1", "2", "3", "4", "5", "6", "7", "8"]
            self.solution_probabilities = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

            for letter in self.row1_labels:
                object_dict = {}

                self.row1[letter] = self.problem.figures[letter]
                self.figures.append(self.problem.figures[letter])
                for k, v in self.problem.figures[letter].objects.items():
                    object_dict[k] = v.attributes
                    self.objects[letter] = object_dict
            for letter in self.row2_labels:
                object_dict = {}

                self.row2[letter] = self.problem.figures[letter]
                self.figures.append(self.problem.figures[letter])
                for k, v in self.problem.figures[letter].objects.items():
                    object_dict[k] = v.attributes
                    self.objects[letter] = object_dict
            for letter in self.row3_labels:
                object_dict = {}

                self.row3[letter] = self.problem.figures[letter]
                self.figures.append(self.problem.figures[letter])
                for k, v in self.problem.figures[letter].objects.items():
                    object_dict[k] = v.attributes
                    self.objects[letter] = object_dict
            for letter in self.solution_labels:
                self.answers[letter] = self.problem.figures[letter]
                self.solutions.append(self.problem.figures[letter])
        else:
            self.row1_labels = ["A", "B"]
            self.row2_labels = ["C"]
            self.solution_labels = ["1", "2", "3", "4", "5", "6"]
            self.solution_probabilities = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

            for letter in self.row1_labels:
                object_dict = {}

                figure = self.problem.figures[letter]

                self.row1[letter] = self.problem.figures[letter]
                self.figures.append(self.problem.figures[letter])

                for k, v in self.problem.figures[letter].objects.items():
                    # obj_label = obj.name
                    # attributes = figure.objects[k].attributes
                    object_dict[k] = v.attributes
                self.objects[letter] = object_dict
            for letter in self.row2_labels:
                object_dict = {}

                figure = self.problem.figures[letter]

                self.row2[letter] = self.problem.figures[letter]
                self.figures.append(self.problem.figures[letter])

                for k,v in self.problem.figures[letter].objects.items():
                    # obj_label = obj.name
                    # object_dict[k] = self.problem.figures[letter].objects[obj].attributes
                    object_dict[k] = v.attributes
                self.objects[letter] = object_dict
            for letter in self.solution_labels:
                self.answers[letter] = self.problem.figures[letter]
                self.solutions.append(self.problem.figures[letter])

    def get_2x2_attributes(self):
        attribute_list = []
        print("self.objects: %s" % str(self.objects))
        for label in self.row1_labels:
            attribute_list.append(self.objects[label])
        for label in self.row2_labels:
            attribute_list.append(self.objects[label])
        return attribute_list

    def get_3x3_attributes(self):
        attribute_list = []
        print("self.objects: %s" % str(self.objects))
        for label in self.row1_labels:
            attribute_list.append(self.objects[label])
        for label in self.row2_labels:
            attribute_list.append(self.objects[label])
        for label in self.row3_labels:
            attribute_list.append(self.objects[label])
        return attribute_list

    def open_images(self, size=2):
        filenames = {}
        prob = self.problem.name.split(" ")
        prob_level = prob[0]
        prob_set = prob[2][0]
        prob_dir = prob_level + " Problems "  + prob_set
        fn_template = "Problems/" + prob_dir + "/" + self.problem.name + "/"
        for label in self.row1_labels:
            k = label
            fn = fn_template + label + ".png"
            img = Image.open(fn)
            img = img.convert('RGB')
            self.images[k] = img
        for label in self.row2_labels:
            k = label
            fn = fn_template + label + ".png"
            img = Image.open(fn)
            img = img.convert('RGB')
            self.images[k] = img
        for label in self.solution_labels:
            k = label
            fn = fn_template + label + ".png"
            img = Image.open(fn)
            img = img.convert('RGB')
            self.images[k] = img
        if size == 3:
            for label in self.row3_labels:
                k = label
                fn = fn_template + label + ".png"
                img = Image.open(fn)
                img = img.convert('RGB')
                self.images[k] = img

    def get_visual_guess_2x2(self):
        pixel_ratio_values = []
        # keys = sorted(self.images.keys())
        solution_pixel_ratios = []
        for k in self.row1_labels:
            image = self.images[k]
            num_blk_pixels = self.get_num_black_pixels(image)
            pixel_ratio_values.append(num_blk_pixels)
        for k in self.row2_labels:
            image = self.images[k]
            num_blk_pixels = self.get_num_black_pixels(image)
            pixel_ratio_values.append(num_blk_pixels)
        for k in self.solution_labels:
            image = self.images[k]
            num_blk_pixels = self.get_num_black_pixels(image)
            solution_pixel_ratios.append(num_blk_pixels)

        horiz_ratio = float(pixel_ratio_values[0]) / float(pixel_ratio_values[1])
        vert_ratio = float(pixel_ratio_values[0]) / float(pixel_ratio_values[2])

        normalized_pixels_horiz = horiz_ratio * pixel_ratio_values[2]
        normalized_pixels_vert = vert_ratio * pixel_ratio_values[1]

        for indx in range(len(solution_pixel_ratios)):
            if normalized_pixels_horiz < solution_pixel_ratios[indx]:
                horiz_prob_ratio = float(normalized_pixels_horiz) / float(solution_pixel_ratios[indx])
            else:
                horiz_prob_ratio = float(solution_pixel_ratios[indx]) / float(normalized_pixels_horiz)
            if  normalized_pixels_vert < solution_pixel_ratios[indx]:
                vert_prob_ratio = float(normalized_pixels_vert) / float(solution_pixel_ratios[indx])
            else:
                vert_prob_ratio = float(solution_pixel_ratios[indx]) / float(normalized_pixels_vert)

            solution_pixel_ratios[indx] *= (vert_prob_ratio*horiz_prob_ratio)

        return solution_pixel_ratios

    def is_black_pixel(self, coords, img, val=20):
        x, y = coords
        r, g, b = img.getpixel((x, y))
        if sum([r, g, b]) < val:
            return True
        else:
            return False

    def get_num_black_pixels(self, img, val=20):
        w, h = img.size
        num_black = 0
        for x in range(w):
            for y in range(h):
                if self.is_black_pixel((x, y), img, val):
                    num_black += 1
        return num_black

    def find_visual_transforms_3x3(self):
        self.calculate_pixel_differences(1)
        self.calculate_pixel_differences(2)
        self.calculate_pixel_differences(3)

        self.calculate_pixel_differences(1, horizontal=False)
        self.calculate_pixel_differences(2, horizontal=False)
        self.calculate_pixel_differences(3, horizontal=False)

        self.calculate_increment_probability()
        self.calculate_no_change_probability()
        self.calculate_position_change_probability()

        for label in self.solution_labels:
            num_blk = self.get_num_black_pixels(self.images[label])
            self.black_pixels_solutions[label] = num_blk

    def calculate_pixel_differences(self, number, horizontal=True):
        labels = []
        num_black_pixels = []

        if horizontal:
            if number == 1:
                labels = self.row1_labels
            elif number == 2:
                labels = self.row2_labels
            elif number == 3:
                labels = self.row3_labels
        else:
            if number == 1:
                labels = self.col1_labels
            elif number == 2:
                labels = self.col2_labels
            elif number == 3:
                labels = self.col3_labels

        for label in labels:
            num_blk = self.get_num_black_pixels(self.images[label])
            num_black_pixels.append(num_blk)
            self.black_pixels_frame[label] = num_blk

        if number != 3:
            most_blk = max(self.black_pixels_frame[labels[0]], self.black_pixels_frame[labels[1]])
            least_blk = min(self.black_pixels_frame[labels[0]], self.black_pixels_frame[labels[1]])
            third_frame_blk = self.black_pixels_frame[labels[2]]

            possible_addition = most_blk + least_blk
            possible_deletion = most_blk - least_blk

            if possible_addition < third_frame_blk:
                addition_probability = float(possible_addition)/float(third_frame_blk)
            else:
                addition_probability = float(third_frame_blk)/float(possible_addition)

            if possible_deletion < third_frame_blk:
                deletion_probability = float(possible_deletion)/float(third_frame_blk)
            else:
                deletion_probability = float(third_frame_blk)/float(possible_deletion)

            if horizontal == True:
                self.horizontal_addition_probability = float(self.horizontal_addition_probability)*addition_probability
                self.horizontal_deletion_probability = float(self.horizontal_deletion_probability)*deletion_probability
            else:
                self.vertical_addition_probability = float(self.vertical_addition_probability)*addition_probability
                self.vertical_deletion_probability = float(self.vertical_deletion_probability)*deletion_probability

    def calculate_increment_probability(self):
        row1_diff = self.calculate_increment(1)
        row2_diff = self.calculate_increment(2)

        col1_diff = self.calculate_increment(1, horizontal=False)
        col2_diff = self.calculate_increment(2, horizontal=False)

        row1_ratio = self.calculate_increment_ratios(row1_diff, 'C')
        row2_ratio = self.calculate_increment_ratios(row2_diff, 'F')

        col1_ratio = self.calculate_increment_ratios(col1_diff, 'G')
        col2_ratio = self.calculate_increment_ratios(col2_diff, 'H')

        self.horizontal_increment_probability = self.horizontal_increment_probability*(row1_ratio*row2_ratio)
        self.vertical_increment_probability = self.vertical_increment_probability*(col1_ratio*col2_ratio)

    def calculate_increment(self, number, horizontal=True):
        if horizontal:
            if number == 1:
                labels = self.row1_labels
            elif number == 2:
                labels = self.row2_labels
        else:
            if number == 1:
                labels = self.col1_labels
            elif number == 2:
                labels = self.col2_labels
        first_frame_val = self.black_pixels_frame[labels[0]]
        second_frame_val = self.black_pixels_frame[labels[1]]

        return (second_frame_val - first_frame_val) + second_frame_val

    def calculate_increment_ratios(self, increment, frame_label):
        # print("self.black_pixels_frame: %s" % str(self.black_pixels_frame))
        num_blk = self.black_pixels_frame[frame_label]
        if increment < num_blk:
            return float(increment)/float(num_blk)
        else:
            return float(num_blk)/float(increment)

    def calculate_no_change_probability(self):
        horizontal_probability = 1
        vertical_probability = 1
        for x in [self.row1_labels, self.row2_labels]:
            frame1 = self.black_pixels_frame[x[0]]
            frame2 = self.black_pixels_frame[x[2]]
            if frame2 < frame1:
                horizontal_probability *= float(frame2)/float(frame1)
            else:
                horizontal_probability *= float(frame1)/float(frame2)
        for x in [self.col1_labels, self.col2_labels]:
            frame1 = self.black_pixels_frame[x[0]]
            frame2 = self.black_pixels_frame[x[2]]
            if frame2 < frame1:
                vertical_probability *= float(frame2)/ float(frame1)
            else:
                vertical_probability *= float(frame1)/ float(frame2)

        self.horizontal_no_change_probability = horizontal_probability
        self.vertical_no_change_probability = vertical_probability

    def calculate_position_change_probability(self):
        first_shift = [self.images["A"], self.images["E"]]
        second_shift = [self.images["B"], self.images["F"]]
        third_shift = [self.images["C"], self.images["D"]]
        fourth_shift = [self.images["D"], self.images["H"]]
        fifth_shift = [self.images["F"], self.images["G"]]

        shifts = [0, 0, 0, 0, 0]
        w, h = self.images['A'].size
        for x in range(w):
            for y in range(h):
                if all(self.is_black_pixel((x, y), img) for img in first_shift):
                    shifts[0] = shifts[0] + 1
                if all(self.is_black_pixel((x, y), img) for img in first_shift):
                    shifts[1] = shifts[1] + 1
                if all(self.is_black_pixel((x, y), img) for img in first_shift):
                    shifts[2] = shifts[2] + 1
                if all(self.is_black_pixel((x, y), img) for img in first_shift):
                    shifts[3] = shifts[3] + 1
                if all(self.is_black_pixel((x, y), img) for img in first_shift):
                    shifts[4] = shifts[4] + 1

        if self.has_black_pixels("A", "E"):
            self.position_change_probability[0] = self.return_shift_ratio(0, shifts, "A", "E")
        if self.has_black_pixels("B", "F"):
            self.position_change_probability[1] = self.return_shift_ratio(1, shifts, "B", "F")
        if self.has_black_pixels("C", "D"):
            self.position_change_probability[2] = self.return_shift_ratio(2, shifts, "C", "D")
        if self.has_black_pixels("D", "H"):
            self.position_change_probability[3] = self.return_shift_ratio(3, shifts, "D", "H")
        if self.has_black_pixels("F", "G"):
            self.position_change_probability[4] = self.return_shift_ratio(4, shifts, "F", "G")

    def has_black_pixels(self, label1, label2):
        if self.black_pixels_frame[label1] > 0 and self.black_pixels_frame[label2] > 0:
            return True
        else:
            return False

    def return_shift_ratio(self, shiftnum, shifts, label1, label2):
        shift_ratios = []
        for label in [label1, label2]:
            shift_ratios.append(float(shifts[shiftnum])/ float(self.black_pixels_frame[label]))
        return min(shift_ratios)

    def calculate_differing_objects(self, number, horizontal=True):
        if horizontal:
            if number == 1:
                labels = self.row1_labels
            elif number == 2:
                labels = self.row2_labels
            elif number == 3:
                labels = self.row3_labels
        else:
            if number == 1:
                labels = self.col1_labels
            elif number == 2:
                labels = self.col2_labels
            elif number == 3:
                labels = self.col3_labels

        object_numbers = []
        for label in labels:
            object_numbers.append(len(self.objects[label].keys()))

        difference = object_numbers[1] - object_numbers[0]
        if object_numbers[2] != difference + object_numbers[1]:
            if horizontal == True:
                self.differing_horiz_objects_probability = self.differing_horiz_objects_probability*0.8
            else:
                self.differing_vert_objects_probability = self.differing_vert_objects_probability*0.8

    def adjust_movement_value(self, moving_possibilities, limit=0.525):
        if self.all_probabilities[-1] >= limit:
            for indx in range(len(self.black_pixels_solutions.keys())):
                if self.position_change_probability[0] < moving_possibilities[indx]:
                    self.solution_probabilities[indx] = self.solution_probabilities[indx]*(self.position_change_probability[0]/float(moving_possibilities[indx]))
                else:
                    self.solution_probabilities[indx] = self.solution_probabilities[indx]*(moving_possibilities[indx]/float(self.position_change_probability[0]))

    def adjust_increment_value(self, limit=0.525):
        if self.horizontal_increment_probability >= limit:
            subtraction = (self.black_pixels_frame["H"] - self.black_pixels_frame["G"]) + self.black_pixels_frame["H"]
            for k, v in self.black_pixels_solutions.items():
                index = int(k) - 1
                if subtraction < v:
                    self.solution_probabilities[index] = self.solution_probabilities[index]*(subtraction/float(v))
                else:
                    self.solution_probabilities[index] = self.solution_probabilities[index]*(v/float(subtraction))
        if self.vertical_increment_probability >= limit:
            subtraction = (self.black_pixels_frame["F"] - self.black_pixels_frame["C"]) + self.black_pixels_frame["F"]
            for k, v in self.black_pixels_solutions.items():
                index = int(k) - 1
                if subtraction < v:
                    self.solution_probabilities[index] = self.solution_probabilities[index]*(subtraction/float(v))
                else:
                    self.solution_probabilities[index] = self.solution_probabilities[index]*(v/float(subtraction))

    def adjust_addition_value(self, limit = 0.525):
        if self.horizontal_addition_probability >= limit:
            addition = (self.black_pixels_frame["H"] + self.black_pixels_frame["G"])
            for k, v in self.black_pixels_solutions.items():
                index = int(k) - 1
                if addition < v:
                    self.solution_probabilities[index] *= addition/float(v)
                else:
                    self.solution_probabilities[index] *= v/float(addition)
        if self.vertical_addition_probability >= limit:
            addition = (self.black_pixels_frame["F"] + self.black_pixels_frame["C"])
            for k, v in self.black_pixels_solutions.items():
                index = int(k) - 1
                if addition < v:
                    self.solution_probabilities[index] *= addition/float(v)
                else:
                    self.solution_probabilities[index] *= v/float(addition)

    def adjust_deletion_value(self, limit = 0.525):
        if self.horizontal_deletion_probability >= limit:
            deletion = max(self.black_pixels_frame["H"], self.black_pixels_frame["G"]) - min(self.black_pixels_frame["H"], self.black_pixels_frame["G"])
            for k, v in self.black_pixels_solutions.items():
                index = int(k) - 1
                if deletion < v:
                    self.solution_probabilities[index] *= deletion/float(v)
                else:
                    self.solution_probabilities[index] *= v/float(deletion)
        if self.vertical_deletion_probability >= limit:
            deletion = max(self.black_pixels_frame["F"], self.black_pixels_frame["C"]) - min(self.black_pixels_frame["F"], self.black_pixels_frame["C"])
            for k, v in self.black_pixels_solutions.items():
                index = int(k) - 1
                if deletion < v:
                    self.solution_probabilities[index] *= deletion/float(v)
                else:
                    self.solution_probabilities[index] *= v/float(deletion)

    def adjust_no_change_value(self, limit = 0.825):
        if self.horizontal_no_change_probability >= limit:
            change_check = self.black_pixels_frame["G"]
            for k, v in self.black_pixels_solutions.items():
                index = int(k) - 1
                if change_check < v:
                    self.solution_probabilities[index] *= change_check/float(v)
                else:
                    self.solution_probabilities[index] *= v/float(change_check)
        if self.vertical_no_change_probability >= limit:
            change_check = self.black_pixels_frame["C"]
            for k, v in self.black_pixels_solutions.items():
                index = int(k) - 1
                if change_check < v:
                    self.solution_probabilities[index] *= change_check/float(v)
                else:
                    self.solution_probabilities[index] *= v/float(change_check)

    def calculate_solution_probabilities(self, limit=0.525):
        self.all_probabilities.append(float(sum(self.position_change_probability))/float(len(self.position_change_probability)))
        movement_possibilities = []
        if all(prob < 0.55 for prob in self.all_probabilities):
            return self.solution_probabilities
        center_image = self.images["E"]

        for solution_label in self.solution_labels:
            solution_image = self.images[solution_label]
            w, h = solution_image.size
            matching_pixels = 0
            for x in range(w):
                for y in range(h):
                    if all(self.is_black_pixel((x, y), img) for img in [solution_image, center_image]):
                        matching_pixels += 1

            solution_black_pixels = self.black_pixels_solutions[solution_label]
            if solution_black_pixels > 0 and self.black_pixels_frame["E"]:
                multiplying_factor = min(matching_pixels/float(solution_black_pixels), matching_pixels/float(self.black_pixels_frame["E"]))
                movement_possibilities.append(1.0*multiplying_factor)
            else:
                movement_possibilities.append(0.0)

        self.adjust_movement_value(movement_possibilities)
        self.adjust_increment_value()
        self.adjust_addition_value()
        self.adjust_deletion_value()
        self.adjust_no_change_value()
