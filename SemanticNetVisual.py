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

        self.solution_labels = []

        # self.file_dir = ""
        self.images = {}

        # self.

        if problem.problemType == "3x3":
            self.row1_labels = ["A", "B", "C"]
            self.row2_labels = ["D", "E", "F"]
            self.row3_labels = ["G", "H"]
            self.solution_labels = ["1", "2", "3", "4", "5", "6", "7", "8"]

            for letter in self.row1_labels:
                object_dict = {}

                self.row1[letter] = self.problem.figures[letter]
                self.figures.append(self.problem.figures[letter])
                # for obj in self.problem.figures[letter].objects:
                for k, v in self.problem.figures[letter].objects.items():
                    # obj_label = obj.name
                    # object_dict[obj_label] = self.problem.figures[letter].objects[obj].attributes
                    object_dict[k] = v.attributes
                    self.objects[letter] = object_dict
            for letter in self.row2_labels:
                object_dict = {}

                self.row2[letter] = self.problem.figures[letter]
                self.figures.append(self.problem.figures[letter])
                # for obj in self.problem.figures[letter].objects:
                for k, v in self.problem.figures[letter].objects.items():
                    # obj_label = obj.name
                    # object_dict[obj_label] = self.problem.figures[letter].objects[obj].attributes
                    object_dict[k] = v.attributes
                    self.objects[letter] = object_dict
            for letter in self.row3_labels:
                object_dict = {}

                self.row3[letter] = self.problem.figures[letter]
                self.figures.append(self.problem.figures[letter])
                # for obj in self.problem.figures[letter].objects:
                for k, v in self.problem.figures[letter].objects.items():
                    # obj_label = obj.name
                    # object_dict[obj_label] = self.problem.figures[letter].objects[obj].attributes
                    object_dict[k] = v.attributes
                    self.objects[letter] = object_dict
            for letter in self.solution_labels:
                self.answers[letter] = self.problem.figures[letter]
                self.solutions.append(self.problem.figures[letter])
        else:
            self.row1_labels = ["A", "B"]
            self.row2_labels = ["C"]
            self.solution_labels = ["1", "2", "3", "4", "5", "6"]

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

    def is_black_pixel(self, coords, img, val=19):
        x, y = coords
        r, g, b = img.getpixel((x, y))
        if sum([r, g, b]) < val:
            return True
        else:
            return False

    def get_num_black_pixels(self, img, val=19):
        w, h = img.size
        num_black = 0
        for x in range(w):
            for y in range(h):
                if self.is_black_pixel((x, y), img, val):
                    num_black += 1
        return num_black
