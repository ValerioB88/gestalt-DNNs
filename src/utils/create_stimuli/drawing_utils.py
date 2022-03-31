import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


class ConstrainedError(Exception):
    pass

class DrawShape():
    def __init__(self, background='black', img_size=(224, 224), width=None, antialiasing=False, borders=False, line_col=None, size_up_size_down=False, borders_width=None, min_dist_bw_points=0, min_dist_borders=0):

        self.min_dist_bw_points = min_dist_bw_points
        self.min_dist_borders = min_dist_borders
        self.size_up_size_down = size_up_size_down
        self.antialiasing = antialiasing
        self.borders = borders
        self.background = background
        self.img_size = img_size
        if width is None:
            width = img_size[0] * 0.022

        # random means random pixels,
        # random uni means random uniform color every creation
        if background == 'random':
            self.background_type = 'rnd-pixels'
            self.line_col = 0 if line_col is None else line_col

        elif background == 'black':
            self.background_type = 'white-on-black'
            self.line_col = 255 if line_col is None else line_col

        elif background == 'white':
            self.background_type = 'black-on-white'
            self.line_col = 0 if line_col is None else line_col

        elif background == 'random_uni':
            self.background_type = 'random_uniform'
            self.line_col = 255 if line_col is None else line_col

        self.fill = (*[self.line_col] * 3, 255)
        self.line_args = dict(fill=self.fill, width=width, joint='curve')
        if borders:
            self.borders_width = self.line_args['width'] if borders_width is None else borders_width
        else:
            self.borders_width = 0


    def get_empty_single(self):
        r = self.line_args['width']

        x0 = np.random.randint(0 + r + self.min_dist_borders + self.borders_width, self.img_size[0] - r - self.borders_width - self.min_dist_borders)
        y0 = np.random.randint(0 + r + self.min_dist_borders + self.borders_width, self.img_size[1] - r - self.borders_width - self.min_dist_borders)
        return (((x0, y0),), ())  # img0-> p1 -> (x, y)

    def get_all_sets(self):
        while True:
            try:
                pp_empty = (), ()
                pp_empty_single = self.get_empty_single()
                pp_single = self.get_pair_points(pp_empty_single)
                pp_proximity = self.get_proximity_points(pp_single)
                pp_orientation = self.get_orientation_points(pp_single)
                pp_linearity = self.get_linearity_o_points(pp_orientation)
                names = ['empty', 'empty-single', 'single', 'proximity', 'orientation', 'linearity']
                pps = [pp_empty, pp_empty_single, pp_single, pp_proximity, pp_orientation, pp_linearity]
                ppsdict = {k: v for k, v in zip(names, pps)}
                sets = {n: (self.draw_all_dots(pp[0]), self.draw_all_dots(pp[1]))  for n, pp in zip(names, pps)}
                return sets, ppsdict

            except ConstrainedError:
                print('Regenerating...')
                continue


    def create_canvas(self, img_size=None, borders=None):
        if img_size is None:
            img_size = self.img_size

        if self.background == 'random':
            img_size = Image.fromarray(np.random.randint(0, 255, (img_size[1], img_size[0], 3)).astype(np.uint8), mode='RGB')
        elif self.background == 'black':
            img_size = Image.new('RGB', tuple(img_size), 'black')  # x and y
        elif self.background == 'white':
            img_size = Image.new('RGB', tuple(img_size), 'white')  # x and y
        elif self.background == 'random_uni':
            img_size = Image.new('RGB', tuple(img_size), (np.random.randint(256), np.random.randint(256), np.random.randint(256)))  # x and y


        borders = self.borders if borders is None else borders
        if borders:
            draw = ImageDraw.Draw(img_size)
            draw.line([(0, 0), (0, img_size.size[0])], fill=self.line_args['fill'], width=self.borders_width)
            draw.line([(0, 0), (img_size.size[0], 0)], fill=self.line_args['fill'], width=self.borders_width)
            draw.line([(img_size.size[0] - 1, 0), (img_size.size[0] - 1, img_size.size[1] - 1)], fill=self.line_args['fill'], width=self.borders_width)
            draw.line([(0, img_size.size[0] - 1), (img_size.size[0] - 1, img_size.size[1] - 1)], fill=self.line_args['fill'], width=self.borders_width)

        return img_size


    def circle(self, draw, center, radius):
        draw.ellipse((center[0] - radius + 1,
                      center[1] - radius + 1,
                      center[0] + radius - 1,
                      center[1] + radius - 1), fill=self.fill, outline=None)

    def apply_antialiasing(self, im):
        im = im.resize((im.size[0] * self.antialiasing, im.size[1] * self.antialiasing))
        im = im.resize((im.size[0] // self.antialiasing, im.size[1] // self.antialiasing), resample=Image.ANTIALIAS)
        return im


    def from_radians_get_line(self, radius, r):
        # radius = (self.img_size[0] - dist_borders) // 2
        return [(radius*np.cos(r), radius*np.sin(r)),
                (-radius * np.cos(r), -radius * np.sin(r))]


    def loc_to_int(self, loc):
        return [tuple([int(i) for i in locc]) for locc in loc]


    def center_at_cavas(self, loc):
        return self.loc_to_int(np.array(loc) + np.array(self.img_size) / 2)


    def draw_parentheses(self, p, d=45, circle_pos='left', draw=None, **kwargs):  # p if x, y
        x, y = p
        r_minor = self.img_size[1]/2 - self.img_size[1]/15
        r_major = self.img_size[0] - self.img_size[0]/3
        if circle_pos == 'left':
            bb = [(-r_major*2 + x, -r_minor + y,),
                  (x,  y + r_minor)]
            draw.arc(bb, -d, d, width=self.line_args['width'], fill=self.fill)

        if circle_pos == 'right':
            bb = [(x, -r_minor + y),
                  (r_major * 2 + x, y + r_minor)]
            draw.arc(bb, 180-d, 180+d,  width=self.line_args['width'], fill=self.fill)
        if circle_pos == 'up':
            bb = [(-r_minor + x, - r_major*2 + y),
                  (r_minor + x, y)]
            draw.arc(bb, 90-d, 90+d,  width=self.line_args['width'], fill=self.fill)
        if circle_pos == 'down':
            bb = [(-r_minor + x, y),
                  (r_minor + x, r_major * 2 +y)]
            draw.arc(bb, 270-d, 270+d, width=self.line_args['width'], fill=self.fill)

    def get_rnd_colon(self, type='h', dist=None):
        repeat = True
        if dist is None:
            dist = self.line_args['width'] * 3
        while repeat:
            r = self.line_args['width']
            # x0 = np.random.uniform(0 + r * 2, self.img_size[0] - r * 2)
            # y0 = np.random.uniform(0 + r * 2, self.img_size[1] - r * 2)

            if type == 'h':
                x0 = self.img_size[0] / 2 - dist / 2
                y0 = self.img_size[1] / 2
                x11 = x0 + dist
                y11 = y0
                # theta = np.random.uniform(0, np.deg2rad(45))
                theta = np.deg2rad(45)
            if type == 'v':
                x0 = self.img_size[0] / 2
                y0 = self.img_size[1] / 2 - dist / 2
                x11 = x0
                y11 = y0 + dist
                # theta = np.random.uniform(np.deg2rad(90),  np.deg2rad(90 + 45))
                theta = np.deg2rad(90 + 45)
            im1 = self.draw_all_dots(((x0, y0), (x11, y11)))

            x12 = dist * np.cos(theta) + x0
            y12 = dist * np.sin(theta) + y0
            im2 = self.draw_all_dots(((x0, y0),(x12, y12)))
            if self.antialiasing:
                im1 = self.apply_antialiasing(im1)
                im2 = self.apply_antialiasing(im2)
            if 0 + r * 2< x11 < self.img_size[0] - r * 2 and 0 + r * 2< y11 < self.img_size[1] - r * 2 \
                    and 0 + r *2 < x12 < self.img_size[0] - r *2 and 0 + r * 2  < y12 < self.img_size[1] - r * 2:
                repeat = False
            else:
                print("Repeating generation")

        return im1, im2

    def get_char_canvas_with_rot(self, im=None, char=';', font_path='C:/windows/Fonts/calibrib.ttf', rot=0, sz_fact=(100, 100), font_size=70):
        if im is None:
            im = self.create_canvas()

        cc = self.create_canvas(img_size=sz_fact, borders=False)
        draw = ImageDraw.Draw(cc)

        arial = ImageFont.truetype(font_path, font_size)
        draw.text((sz_fact[0] / 2, sz_fact[1] / 2), char, font=arial, anchor="mm", fill=(self.line_col,)*3)
        cc = cc.rotate(rot, resample=2, center=(sz_fact[0] // 2, sz_fact[1] // 2))
        # plt.imshow(cc)

        # cc.show()
        im.paste(cc, (self.img_size[0] // 2 - sz_fact[0] // 2,
                      self.img_size[1] // 2 - sz_fact[1] // 2,
                      self.img_size[0] // 2 - sz_fact[0] // 2 + sz_fact[0],
                      self.img_size[1] // 2 - sz_fact[1] // 2 + sz_fact[1]))
        # im1.show()
        return im

    def get_rnd_char_rotated(self, char, type='h'):
        if type == 'h':
            im1 = self.get_char_canvas_with_rot(char, 90)
            im2 = self.get_char_canvas_with_rot(char, 90+45)
        if type == 'v':
            im1 = self.get_char_canvas_with_rot(char, 0)
            im2 = self.get_char_canvas_with_rot(char, 360-45)
        return im1, im2

    def get_beak(self, type):
        im1, im2 = self.get_rnd_char_rotated(';' ,'v')
        draw = ImageDraw.Draw(im1)
        draw.arc([(self.img_size[0]//2 - 35, self.img_size[1]//2 - 30),
                  (self.img_size[0]//2 + 10, self.img_size[1]//2 + 30)], 0, 360)
        draw = ImageDraw.Draw(im2)
        draw.arc([(self.img_size[0] // 2 - 35, self.img_size[1] // 2 - 30),
                  (self.img_size[0] // 2 + 10, self.img_size[1] // 2 + 30)], 0, 360)
        return im1, im2

    def add_circles_to_loc(self, l, draw):
        self.circle(draw, l[0], self.line_args['width'] // 2)
        self.circle(draw, l[1], self.line_args['width'] // 2)
    ##
    def get_triangle(self, type='base_0'):
        im = self.create_canvas()
        draw = ImageDraw.Draw(im)
        if type == 'base_0':
            t = 45 + 90
            radius = np.round(np.sqrt((im.size[0] * 1 // 4) * (im.size[0] * 1 // 4) + (im.size[0] * 1 // 4) * (im.size[0] * 1 // 4)))
            l = self.center_at_cavas(self.from_radians_get_line(radius, t / 360 * (2 * np.pi)))
            draw = ImageDraw.Draw(im)
            draw.line(l, **self.line_args)
            self.add_circles_to_loc(l, draw)
        if type == 'base_1':
            t = 45
            radius = np.round(np.sqrt((im.size[0] * 1 // 4) * (im.size[0] * 1 // 4) + (im.size[0] * 1 // 4) * (im.size[0] * 1 // 4)))
            l = self.center_at_cavas(self.from_radians_get_line(radius, t / 360 * (2 * np.pi)))
            draw = ImageDraw.Draw(im)
            draw.line(l, **self.line_args)
            self.add_circles_to_loc(l, draw)

        if type == 'composite_0':
            t = 45 + 90
            radius = np.round(np.sqrt((im.size[0] * 1 // 4) * (im.size[0] * 1 // 4) + (im.size[0] * 1 // 4) * (im.size[0] * 1 // 4)))
            l = self.center_at_cavas(self.from_radians_get_line(radius, t / 360 * (2 * np.pi)))
            draw.line(l, **self.line_args)
            self.add_circles_to_loc(l, draw)

            vert_l = [(im.size[0] * 1 // 4, im.size[0] * 1 // 4), (im.size[0] * 1 // 4, im.size[0] * 3 // 4)]
            draw.line(vert_l, **self.line_args)
            self.add_circles_to_loc(vert_l, draw)

            vert_l = [(im.size[0] * 1 // 4, im.size[0] * 3 // 4), (im.size[0] * 3 // 4, im.size[0] * 3 // 4)]
            draw.line(vert_l, **self.line_args)
            self.add_circles_to_loc(vert_l, draw)

        if type == 'composite_1':
            t = 45
            radius = np.round(np.sqrt((im.size[0] * 1 // 4) * (im.size[0] * 1 // 4) + (im.size[0] * 1 // 4) * (im.size[0] * 1 // 4)))
            l = self.center_at_cavas(self.from_radians_get_line(radius, t / 360 * (2 * np.pi)))
            draw.line(l, **self.line_args)
            self.add_circles_to_loc(l, draw)

            vert_l = [(im.size[0] * 1 // 4, im.size[0] * 1 // 4), (im.size[0] * 1 // 4, im.size[0] * 3 // 4)]
            draw.line(vert_l, **self.line_args)
            self.add_circles_to_loc(vert_l, draw)

            vert_l = [(im.size[0] * 1 // 4, im.size[0] * 3 // 4), (im.size[0] * 3 // 4, im.size[0] * 3 // 4)]
            draw.line(vert_l, **self.line_args)
            self.add_circles_to_loc(vert_l, draw)
        return im

    def get_pair_points(self, one_point=None):
        repeat = True
        while repeat:
            r = self.line_args['width']
            if not one_point:
                x0 = np.random.randint(0 + r + self.borders_width + self.min_dist_borders, self.img_size[0] - r - self.borders_width - self.min_dist_borders)
                y0 = np.random.randint(0 + r + self.borders_width + self.min_dist_borders, self.img_size[1] - r - self.borders_width - self.min_dist_borders)
            else:
                (((x0, y0),), _) = one_point

            x1 = np.random.randint(0 + r + self.min_dist_borders + self.borders_width, self.img_size[0] - r - self.borders_width - self.min_dist_borders)
            y1 = np.random.randint(0 + r + self.min_dist_borders + self.borders_width, self.img_size[1] - r - self.borders_width - self.min_dist_borders)

            if np.linalg.norm([np.array([x0, y0]) - np.array([x1, y1])]) > self.min_dist_bw_points:
                repeat = False
            else:
                print("Repeat min distance pair points")

        return ((x0, y0),), ((x1, y1),)  # img0-> p1 -> (x, y)

    def get_proximity_points(self, pp=None, dist=None, **kwargs):
        r = self.line_args['width']
        stop = False
        count = 0
        while not stop:

            if pp is None:
                ((x0, y0),), ((x1, y1),) = self.get_pair_points(**kwargs)
            else:
                ((x0, y0),), ((x1, y1),) = pp

            A = np.array([x0, y0])
            B = np.array([x1, y1])
            Diff = (A - B)
            if dist is None:
                dist = np.random.uniform((r * 2) / (np.linalg.norm(A - B)), 0.4)
            else:
                stop = True ## if dist is given, we don't check for any constrains
            L = A + dist * (B - A)
            xx_lin, yy_lin = int(L[0]), int(L[1])
            if np.linalg.norm([xx_lin, yy_lin] - A) > r * 2 and np.linalg.norm([xx_lin, yy_lin] - B) > r * 2 \
                    and  self.img_size[0] - r - self.borders_width - self.min_dist_borders > xx_lin > 0 + r + self.borders_width + self.min_dist_borders < yy_lin < self.img_size[1] - r - self.borders_width - self.min_dist_borders:
                stop = True
                # self.plot_all_points([((x0, y0), (xx_lin, yy_lin)), ((x1, y1), (xx_lin, yy_lin))])
                # self.plot_all_points([((x0, y0),), ((x1, y1),)])
            count += 1
            if count > 100:
                raise ConstrainedError("Can't generate proximity points")

        return ((x0, y0), (xx_lin, yy_lin)), ((x1, y1), (xx_lin, yy_lin)),

    def get_orientation_points(self, pp=None, **kwargs):
        count = 0
        stop = False
        while not stop:
            r = self.line_args['width']
            if pp is None:
                ((x0, y0),), ((x1, y1),) = self.get_pair_points(**kwargs)
            else:
                ((x0, y0),), ((x1, y1),) = pp

            diagonal = np.sqrt(self.img_size[0] ** 2 + self.img_size[1] ** 2)
            distance = np.linalg.norm(np.array([x0, y0]) - np.array([x1, y1]))
            radius_circles = np.random.uniform(np.max([r, distance / 2]), diagonal * 0.7)

            xor0, yor0, xor1, yor1 = self.intersections(x0, y0, radius_circles, x1, y1, radius_circles)

            if np.random.randint(1) == 0:
                xx_equi, yy_equi = xor0, yor0
            else:
                xx_equi, yy_equi = xor1, yor1

            if 0 + r +self.borders_width + self.min_dist_borders< xx_equi < self.img_size[0] - r - self.borders_width  - self.min_dist_borders and 0 + r + self.borders_width + self.min_dist_borders< yy_equi < self.img_size[1] - r - self.borders_width - self.min_dist_borders:
                stop = True
            count += 1
            if count > 20:
                raise ConstrainedError("Can't generate orientation points")

        return ((x0, y0), (xx_equi, yy_equi)), ((x1, y1), (xx_equi, yy_equi))

    def plot_all_points(self, pps):
        ims = [self.draw_all_dots(pp) for idx, pp in enumerate(pps)]
        self.plot_all_imgs(ims)

    def plot_all_imgs(self, im):
        fig, ax = plt.subplots(2, int(np.ceil(len(im) / 2)))
        ax = np.array([ax]) if len(im) == 1 else ax.flatten()
        [i.axis('off') for i in ax.flatten()]
        for idx, i in enumerate(im):
            ax[idx].imshow(i)


    def get_linearity_o_points(self, pp=None, **kwargs):
        r = self.line_args['width']
        count = 0
        stop = False
        while not stop:
            r = self.line_args['width']
            if pp is None:
                ((x0, y0), (xor, yor)), ((x1, y1), (xor, yor)) = self.get_orientation_points(**kwargs)
            else:
                ((x0, y0), (xor, yor)), ((x1, y1), (xor, yor)) = pp

            A = np.array([x0, y0])
            B = np.array([xor, yor])
            Diff = (A - B)
            s = Diff[1] / Diff[0]

            dd = np.random.uniform((r*2)/(np.linalg.norm(A-B)), 0.4)
            L = B + dd*(A-B)
            xx_lin, yy_lin = int(L[0]), int(L[1])
            # plt.figure(1)
            # plt.imshow(self.draw_all_dots((((x0, y0), (xor, yor), (L[0], L[1])))))
            # plt.figure(2)
            # plt.imshow(self.draw_all_dots((((x1, y1), (xor, yor), (L[0], L[1])))))


            # if round(A[0] - B[0]) == 0 or round(A[1] - B[1]) == 0:
            #     raise ConstrainedError("Can't generate linearity/o points")
            ## OLD LINEARITY

            # xx_lin = np.random.randint(
            #     np.min([np.min([np.clip(A[0] - A[1] / s, r, self.img_size[1] - r)]),
            #             np.max([np.clip((self.img_size[1] - r - A[1]) / s + A[0], r, self.img_size[1]- r)])]),
            #
            #     np.max([np.min([np.clip(A[0] - A[1] / s, r, self.img_size[1] - r)]),
            #             np.max([np.clip((self.img_size[1] - r - A[1]) / s + A[0], r, self.img_size[1] - r)])]))
            #
            # yy_lin = int(s * (xx_lin - A[0]) + A[1])
            #

            if np.linalg.norm([xx_lin, yy_lin] - A) > r * 2 \
                    and np.linalg.norm([xx_lin, yy_lin] - B) > r * 2 \
                    and self.img_size[0] - r - self.borders_width + self.min_dist_borders> xx_lin > 0 + r + self.borders_width + self.min_dist_borders < yy_lin < self.img_size[1] - r - self.borders_width - self.min_dist_borders:
                stop = True
            count += 1
            if count > 100:
                raise ConstrainedError("Can't generate linearity/o points")
            # p = ((x0, y0), (xor, yor), (xx_lin, yy_lin)), ((x1, y1), (xor, yor), (xx_lin, yy_lin))
            # im = {}
            # im['prova'] = self.draw_all_dots(p)
            # self.plot_all_imgs(im)

        return ((x0, y0), (xor, yor), (xx_lin, yy_lin)), ((x1, y1), (xor, yor), (xx_lin, yy_lin))

    def draw_set(self, pps):
        r = self.line_args['width']
        c = 1
        images_set = []
        for idx, s in enumerate(pps[0]):
            images = []
            for im_pp in pps:
                im = self.create_canvas()
                draw = ImageDraw.Draw(im)
                for p in im_pp[0:idx+1]:
                    self.circle(draw, (p[0], p[1]), radius=r)
                images.append(im)
            images_set.append(images)
        return images_set


    def draw_all_dots(self, pps):
        r = self.line_args['width']
        im = self.create_canvas()
        draw = ImageDraw.Draw(im)
        for p in pps:
            self.circle(draw, (p[0], p[1]), radius=r)
        if self.antialiasing:
            im = self.apply_antialiasing(im)

        return im


    def intersections(self, x0, y0, r0, x1, y1, r1):
        import math
        # circle 1: (x0, y0), radius r0
        # circle 2: (x1, y1), radius r1

        d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # non intersecting
        if d > r0 + r1:
            return {}
        # One circle within other
        if d < abs(r0 - r1):
            return {}
        # coincident circles
        if d == 0 and r0 == r1:
            return {}
        else:
            a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
            h = math.sqrt(r0 ** 2 - a ** 2)
            x2 = x0 + a * (x1 - x0) / d
            y2 = y0 + a * (y1 - y0) / d
            x3 = x2 + h * (y1 - y0) / d
            y3 = y2 - h * (x1 - x0) / d
            x4 = x2 - h * (y1 - y0) / d
            y4 = y2 + h * (x1 - x0) / d
            return int(x3), int(y3), int(x4), int(y4)


    def two_points_and_equidistance_point(self):
        stop = False
        while not stop:
            r = self.line_args['width']

            x_base_0 = np.random.randint(0 + r, self.img_size[0] - r)
            y_base_0 = np.random.randint(0 + r, self.img_size[1] - r)

            x_base_1 = np.random.randint(0 + r, self.img_size[0] - r)
            y_base_1 = np.random.randint(0 + r, self.img_size[1] - r)

            diagonal = np.sqrt(self.img_size[0] ** 2 + self.img_size[1] ** 2)
            distance = np.linalg.norm(np.array([x_base_0, y_base_0]) - np.array([x_base_1, y_base_1]))
            radius_circles = np.random.uniform(distance / 2, diagonal * 0.7)

            x1, y1, x2, y2 = self.intersections(x_base_0, y_base_0, radius_circles, x_base_1, y_base_1, radius_circles)

            if np.random.randint(1) == 0:
                xx_equi, yy_equi = x1, y1
            else:
                xx_equi, yy_equi = x2, y2

            if 0 + r < xx_equi < self.img_size[0] - r and 0 + r < yy_equi < self.img_size[1] - r:
                stop = True

        return x_base_0, y_base_0, x_base_1, y_base_1, xx_equi, yy_equi


    def get_rnd_symmetry_p(self):
        r = self.line_args['width']

        x_base_0, y_base_0, x_base_1, y_base_1, xx_equi, yy_equi = self.two_points_and_equidistance_point()

        symm_point_ax = self.img_size[0] - x_base_0
        symm_point_bx = self.img_size[1] - xx_equi
        im_base_symm1 = self.create_canvas()
        draw = ImageDraw.Draw(im_base_symm1)
        self.circle(draw, (x_base_0, y_base_0), radius=r)
        self.circle(draw, (xx_equi, yy_equi), radius=r)
        self.circle(draw, (symm_point_ax, y_base_0), radius=r)
        self.circle(draw, (symm_point_bx, yy_equi), radius=r)
        # im_comp_symm.show()

        symm_point_ax = self.img_size[0] - x_base_0
        symm_point_bx = self.img_size[1] - xx_equi
        im_base_symm2 = self.create_canvas()
        jitter = lambda: np.random.randint(-self.img_size[0] / 10, self.img_size[0] / 10)
        draw = ImageDraw.Draw(im_base_symm2)
        self.circle(draw, (x_base_0, y_base_0), radius=r)
        self.circle(draw, (xx_equi, yy_equi), radius=r)
        self.circle(draw, (np.clip(symm_point_ax + jitter(), 0 + r, self.img_size[0] - r),
                           np.clip(y_base_0 + jitter(), 0 + r, self.img_size[0] - r)), radius=r)
        self.circle(draw, (np.clip(symm_point_bx + jitter(), 0 + r, self.img_size[0] - r),
                           np.clip(yy_equi + jitter(), 0 + r, self.img_size[0] - r)), radius=r)
        # im_comp_asymm.show()
        return im_base_symm1, im_base_symm2

    def get_rnd_symmetry_p_set(self):
        r = self.line_args['width']

        x_base_0, y_base_0, x_base_1, y_base_1, xx_equi, yy_equi = self.two_points_and_equidistance_point()
        im_base_0 = self.create_canvas()
        draw = ImageDraw.Draw(im_base_0)
        self.circle(draw, (x_base_0, y_base_0), radius=r)
        self.circle(draw, (xx_equi, yy_equi), radius=r)
        if self.antialiasing:
            im_base_0 = self.apply_antialiasing(im_base_0)

        im_base_1 = self.create_canvas()
        draw = ImageDraw.Draw(im_base_1)
        self.circle(draw, (x_base_1, y_base_1), radius=r)
        self.circle(draw, (xx_equi, yy_equi), radius=r)
        if self.antialiasing:
            im_base_1 = self.apply_antialiasing(im_base_1)

        symm_point_ax = self.img_size[0] - x_base_0
        symm_point_bx = self.img_size[1] - xx_equi
        im_comp_symm = self.create_canvas()
        draw = ImageDraw.Draw(im_comp_symm)
        self.circle(draw, (x_base_0, y_base_0), radius=r)
        self.circle(draw, (xx_equi, yy_equi), radius=r)
        self.circle(draw, (symm_point_ax, y_base_0), radius=r)
        self.circle(draw, (symm_point_bx, yy_equi), radius=r)
        # im_comp_symm.show()

        symm_point_ax = self.img_size[0] - x_base_0
        symm_point_bx = self.img_size[1] - xx_equi
        im_comp_asymm = self.create_canvas()
        jitter = lambda: np.random.randint(-self.img_size[0] / 10, self.img_size[0] / 10)
        draw = ImageDraw.Draw(im_comp_asymm)
        self.circle(draw, (x_base_0, y_base_0), radius=r)
        self.circle(draw, (xx_equi, yy_equi), radius=r)
        self.circle(draw, (np.clip(symm_point_ax + jitter(), 0+r, self.img_size[0]-r) ,
                           np.clip(y_base_0 + jitter(), 0+r, self.img_size[0]-r)), radius=r)
        self.circle(draw,  (np.clip(symm_point_bx + jitter(), 0+r, self.img_size[0]-r) ,
                            np.clip(yy_equi + jitter(), 0+r, self.img_size[0]-r)), radius=r)
        # im_comp_asymm.show()
        return im_base_0, im_base_1, im_comp_symm, im_comp_asymm


    def get_parentheses_crossed_set(self, type, offset=0):
        im = self.create_canvas()
        draw = ImageDraw.Draw(im)
        if type == 'base_0':
            p = [self.img_size[0] / 2 - self.img_size[0]*0.09, self.img_size[1] / 2]
            self.draw_parentheses(p, 35, 'right', draw=draw)
            if self.antialiasing:
                im = self.apply_antialiasing(im)
        if type == 'base_1':
            p = [self.img_size[0] / 2 + self.img_size[0]*0.09, self.img_size[1] / 2]
            self.draw_parentheses(p, 35, 'left', draw=draw)
            if self.antialiasing:
                im = self.apply_antialiasing(im)
        if type == 'composite_0':
            p = [self.img_size[0] / 2 + self.img_size[0]*offset - self.img_size[0]*0.09, self.img_size[1] / 2]
            self.draw_parentheses(p, 35, 'right', draw=draw)
            p = [self.img_size[0] / 2 - self.img_size[0]*offset, self.img_size[1] / 2 + 20]
            self.draw_parentheses(p, 35, 'up', draw=draw)
            if self.antialiasing:
                im = self.apply_antialiasing(im)
        if type == 'composite_1':
            p = [self.img_size[0] / 2 + self.img_size[0]*offset + self.img_size[0]*0.09, self.img_size[1] / 2]
            self.draw_parentheses(p, 35, 'left', draw=draw)
            p = [self.img_size[0] / 2 - self.img_size[0]*offset, self.img_size[1] / 2 + 20]
            self.draw_parentheses(p, 35, 'up', draw=draw)
            if self.antialiasing:
                im = self.apply_antialiasing(im)
        return im


    def get_parentheses_set(self, type='base_0', dist_factor=10):
        im = self.create_canvas()
        draw = ImageDraw.Draw(im)
        if type == 'base_0':
            p = [self.img_size[0] / 2 - self.img_size[0]*0.09, self.img_size[1] / 2]
            self.draw_parentheses(p, 35, 'right', draw=draw)
            if self.antialiasing:
                im = self.apply_antialiasing(im)
        if type == 'base_1':
            p = [self.img_size[0] / 2 + self.img_size[0]*0.09, self.img_size[1] / 2]
            self.draw_parentheses(p, 35, 'left', draw=draw)
            if self.antialiasing:
                im = self.apply_antialiasing(im)
        if type == 'composite_0':
            p = [self.img_size[0] / 2 - self.img_size[0] / dist_factor - self.img_size[0]*0.09, self.img_size[1] / 2]
            self.draw_parentheses(p, 35, 'right', draw=draw)
            p = [self.img_size[0] / 2 + self.img_size[0] / dist_factor - self.img_size[0]*0.067, self.img_size[1] / 2]
            self.draw_parentheses(p, 35, 'right', draw=draw)
            if self.antialiasing:
                im = self.apply_antialiasing(im)
        if type == 'composite_1':
            p = [self.img_size[0] / 2 - self.img_size[0] / dist_factor - self.img_size[0]*0.09, self.img_size[1] / 2]
            self.draw_parentheses(p, 35, 'right', draw=draw)
            p = [self.img_size[0] / 2 + self.img_size[0] / dist_factor + self.img_size[0]*0.09, self.img_size[1] / 2]
            self.draw_parentheses(p, 35, 'left', draw=draw)
            if self.antialiasing:
                im = self.apply_antialiasing(im)
        return im

    def draw_line_with_circled_edge(self, draw, xy, **kwargs):
        draw.line(xy, **self.line_args)
        self.add_circles_to_loc(xy, draw)

    def resize_up_down(fun):
        def wrap(self, *args, **kwargs):
            ori_self_width = self.line_args['width']
            self.line_args['width'] = self.line_args['width'] * 2
            original_img_size = self.img_size
            self.img_size = np.array(self.img_size) * 2

            im1, im2 = fun(self, *args, **kwargs)
            im1 = im1.resize(original_img_size)
            im2 = im2.resize(original_img_size)
            self.line_args['width'] = ori_self_width
            self.img_size = original_img_size
            return im1, im2
        return wrap

    @resize_up_down
    def get_array1(self):
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        s = 7
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] //2 - (self.img_size[0] //2 // s), self.img_size[1] //2 -(self.img_size[0] //2 // s)),
                                                (self.img_size[0] //2 - (self.img_size[0] //2 // s), self.img_size[1] //2 + (self.img_size[0] //2 // s))))
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s), self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s), self.img_size[1] // 2 - (self.img_size[0] // 2 // s))))

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // 2 // s), self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s), self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))
        self.draw_line_with_circled_edge(draw,((self.img_size[0] // 2 - (self.img_size[0] // 2 // s), self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                               (self.img_size[0] // 2 + (self.img_size[0] // 2 // s), self.img_size[1] // 2 - (self.img_size[0] // 2 // s))))
        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_array2(self):
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        s = 2.5  # increase this to reduce general size of the square
        scale_hole = 6  # inrease this to reduce hole in the square
        ## top left
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] //2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] //2 // s)),

                                                (self.img_size[0] // 2 - (self.img_size[0] //2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] //scale_hole // s))),
                                         width=self.line_args['width'])


        self.draw_line_with_circled_edge(draw,((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                               (self.img_size[0] // 2 - (self.img_size[0] // scale_hole // s),
                                                self.img_size[1] // 2 - (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])

        ## top right
        self.draw_line_with_circled_edge(draw,((self.img_size[0] // 2 + (self.img_size[0] // scale_hole // s),
                                                self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),

                                               (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                self.img_size[1] // 2 - (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // scale_hole // s))),
                                         width=self.line_args['width'])

        ## down right
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // scale_hole // s)),

                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // scale_hole // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),

                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])

        ## down left
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),

                                                (self.img_size[0] // 2 - (self.img_size[0] // scale_hole // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // scale_hole // s)),

                                                (self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])

        ########################
        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        ## top left
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),

                                                (self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // scale_hole // s))),
                                         width=self.line_args['width'])

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 - (self.img_size[0] // 2 // s) -
                                                 ((self.img_size[0] // 2 -
                                                   (self.img_size[0] // scale_hole // s)) -
                                                  (self.img_size[0] // 2 - (self.img_size[0] // 2 // s))),

                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])

        ## top right
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // scale_hole // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),

                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // scale_hole // s))),
                                         width=self.line_args['width'])

        ## down right
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // scale_hole // s)),

                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // scale_hole // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),

                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])


        ## down left
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),

                                                (self.img_size[0] // 2 - (self.img_size[0] // scale_hole // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // scale_hole // s)),

                                                (self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        return im1, im2

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        s = 2.5
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // 2 // s), self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s), self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s), self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s), self.img_size[1] // 2 - (self.img_size[0] // 2 // s))))
        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_array3(self):
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        s = 2.5
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))
        self.circle(draw, ( self.img_size[0] // 2, self.img_size[0] // 2), self.line_args['width']*0.65)

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + self.img_size[0] // 2 // s, self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s, self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))
        self.circle(draw, ( self.img_size[0] // 2, self.img_size[0] // 2), self.line_args['width']*0.65)

        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_array4(self):
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        s = 2.5
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[0] // 2 + self.img_size[0] // 2 // s),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s, self.img_size[0] // 2 + self.img_size[0] // 2 // s)))
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[0] // 2 - self.img_size[0] // 2 // s),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s, self.img_size[0] // 2 + self.img_size[0] // 2 // s)))

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s, self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + self.img_size[0] // 2 // s, self.img_size[0] // 2 + self.img_size[0] // 2 // s),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s, self.img_size[0] // 2 - self.img_size[0] // 2 // s)))
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[0] // 2 - self.img_size[0] // 2 // s),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s, self.img_size[0] // 2 + self.img_size[0] // 2 // s)))

        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_array5(self):
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        s = 3
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s), self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 - (self.img_size[0] // 2 // s), self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s), self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s), self.img_size[1] // 2 - (self.img_size[0] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2, self.img_size[0] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s,
                                                 self.img_size[0] // 2 - self.img_size[0] // (s * 0.8))))

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // 2 // s), self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s), self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s), self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s), self.img_size[1] // 2 - (self.img_size[0] // 2 // s))))
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2, self.img_size[0] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s,
                                                 self.img_size[0] // 2 - self.img_size[0] // (s * 0.8))))


        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_array6(self):
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        s = 2.5
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] //2, self.img_size[1] //2 -(self.img_size[0] //2 // s)),
                                                (self.img_size[0] //2, self.img_size[1] //2 + (self.img_size[0] //2 // s))))

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[1] // 2 // s, self.img_size[1] // 2),
                                                (self.img_size[0] // 2 + self.img_size[1] // 2 // s, self.img_size[1] // 2)))
        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_array7(self):
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        s = 2.5
        ## vertical line
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2,
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2,
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2,
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2)))

        self.draw_line_with_circled_edge(draw, (((self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                  self.img_size[1] // 2)),
                                                (self.img_size[0] // 2,
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        ## vertical line
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2)))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2,
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2)))

        self.draw_line_with_circled_edge(draw, (((self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                  self.img_size[1] // 2)),
                                                (self.img_size[0] // 2,
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_array8(self):
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        s = 2.5
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2,
                                                 self.img_size[1] // 2),
                                                (self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 - (self.img_size[0] // 2 // s) -  (self.img_size[0] // 4 // s),
                                                 (self.img_size[0] // 2 - (self.img_size[0] // 2 // s) + (self.img_size[0] // 4 // s)))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2, self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2, self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2,
                                                 self.img_size[1] // 2),
                                                (self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 - (self.img_size[0] // 2 // s) - (self.img_size[0] // 4 // s),
                                                 (self.img_size[0] // 2 - (self.img_size[0] // 2 // s) + (self.img_size[0] // 4 // s)))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[1] // 2 // s, self.img_size[1] // 2),
                                                (self.img_size[0] // 2 + self.img_size[1] // 2 // s, self.img_size[1] // 2)))

        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2


    def get_square(self, s):
        im = self.create_canvas()
        draw = ImageDraw.Draw(im)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s))))
        return im

    @resize_up_down
    def get_array9(self):
        s = 2.5
        im1 = self.get_square(s=s)
        draw = ImageDraw.Draw(im1)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0]/2,self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0]/2,self.img_size[1] // 2 + (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])

        im2 = self.get_square(s=s)
        draw = ImageDraw.Draw(im2)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] / 2 - (self.img_size[0] // 2 // s), self.img_size[1] // 2),
                                                (self.img_size[0] / 2 + (self.img_size[0] // 2 // s), self.img_size[1] // 2)),
                                         width=self.line_args['width'])

        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_array10(self):
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        s = 2.5
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s))))

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))
        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_array11(self):
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        s = 2.5
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[0] // 2 + self.img_size[0] // 2 // s),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s, self.img_size[0] // 2 + self.img_size[0] // 2 // s)))
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[0] // 2 + self.img_size[0] // 2 // s),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s, self.img_size[0] // 2 - self.img_size[0] // 2 // s)))

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[0] // 2 + self.img_size[0] // 2 // s),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s, self.img_size[0] // 2 + self.img_size[0] // 2 // s)))
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[0] // 2 - self.img_size[0] // 2 // s),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s, self.img_size[0] // 2 + self.img_size[0] // 2 // s)))

        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    #

    @resize_up_down
    def get_array12(self):
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        s = 2.5
        s2 = 2
        s3 = 1.25
        self.draw_line_with_circled_edge(draw, ((self.img_size[0]//2, self.img_size[1] // 2),
                                                (self.img_size[0]//2, self.img_size[1] // 2 - (self.img_size[0] // s3 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2,  self.img_size[1] // 2 - (self.img_size[0] // s3 // s)),
                                                (self.img_size[1] // 2 - (self.img_size[0] // s2 // s), self.img_size[1] // 2 - (self.img_size[0] // s3 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s))))

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2, self.img_size[1] // 2),
                                                (self.img_size[0] // 2, self.img_size[1] // 2 - (self.img_size[0] // s3 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2, self.img_size[1] // 2 - (self.img_size[0] // s3 // s)),
                                                (self.img_size[1] // 2 - (self.img_size[0] // s2 // s), self.img_size[1] // 2 - (self.img_size[0] // s3 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))
        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_array13(self):
        s = 2.5
        im1 = self.get_square(s)
        draw = ImageDraw.Draw(im1)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s))))

        im2 = self.get_square(s)
        draw = ImageDraw.Draw(im2)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))
        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_array14(self):
        s = 2.5
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0]// 2,
                                                 self.img_size[1] // 2 - (self.img_size[1] // 2 // s)),
                                                (self.img_size[0] //2 ,
                                                 self.img_size[1] // 2 + (self.img_size[1] // 2 // s))))

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2, self.img_size[1] // 2 - (self.img_size[1] // 2 // (s * 2))),
                                                (self.img_size[0] // 2, self.img_size[1] // 2 + (self.img_size[1] // 2 // (s * 2)))))
        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_array15(self):
        s = 2.5
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2,
                                                 self.img_size[1] // 2 - (self.img_size[1] // 2 // s)),
                                                (self.img_size[0] // 2,
                                                 self.img_size[1] // 2 + (self.img_size[1] // 2 // s))))
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2, self.img_size[0] // 2),
                                                (self.img_size[0] // 2 + (self.img_size[1] // 2 // s), self.img_size[0] // 2)),
                                         width=self.line_args['width'])

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0]//2,self.img_size[0]//2 ),
                                                (self.img_size[0]//2, self.img_size[0]//2 +  (self.img_size[1] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2, self.img_size[0] // 2),
                                                (self.img_size[0] // 2 + (self.img_size[1] // 2 // s), self.img_size[0] // 2)),
                                         width=self.line_args['width'])
        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_array16(self):
        s = 2.5
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0]//2 - (self.img_size[1] // 2 // s), self.img_size[0]//2 + (self.img_size[1] // 2 // s)),
                                                (self.img_size[0]//2 + (self.img_size[1] // 2 // s), self.img_size[0]//2 + (self.img_size[1] // 2 // s) )))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2, self.img_size[0] // 2 + (self.img_size[1] // 2 // s)),
                                                (self.img_size[0] // 2, self.img_size[0] // 2 - (self.img_size[1] // 2 // s))),
                                         width=self.line_args['width'])

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[1] // 2 // s), self.img_size[0] // 2 + (self.img_size[1] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[1] // 2 // s), self.img_size[0] // 2 + (self.img_size[1] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2, self.img_size[0] // 2 + (self.img_size[1] // 2 // s)),
                                                (self.img_size[0] // 2, self.img_size[0] // 2)),
                                         width=self.line_args['width'])

        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_array17(self):
        s = 2.5
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] //2 , self.img_size[1] //2   + (self.img_size[1] // 2 // s)),
                                                (self.img_size[0] //2  + (self.img_size[1] // 2 // s) ,  self.img_size[1] //2   + (self.img_size[1] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2, self.img_size[1] //2   + (self.img_size[1] // 2 // s)),
                                                (self.img_size[0] // 2,  self.img_size[1] //2  - (self.img_size[1] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[1] // 2 // s), self.img_size[1] // 2 + (self.img_size[1] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[1] // 2 // s), self.img_size[1] // 2)))

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2, self.img_size[1] // 2 + (self.img_size[1] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[1] // 2 // s), self.img_size[1] // 2 + (self.img_size[1] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 , self.img_size[1] // 2 + (self.img_size[1] // 2 // s)),
                                                (self.img_size[0] // 2 , self.img_size[1] // 2)))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[1] // 2 // s), self.img_size[1] // 2 + (self.img_size[1] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[1] // 2 // s), self.img_size[1] // 2)))

        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_array18(self):
        s = 2.5
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        draw.polygon(((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                       self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),

                      (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                       self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),

                      (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                       self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),


                      (self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                       self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),

                      (self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                       self.img_size[1] // 2 - (self.img_size[0] // 2 // s))), fill=self.line_args['fill'])

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        self.circle(draw, (self.img_size[0]//2,self.img_size[1]//2), radius=self.line_args['width'])

        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_arrayA(self):
        im1 = self.get_parentheses_crossed_set(type='base_0')
        im2 = self.get_parentheses_crossed_set(type='base_1')
        return im1, im2


    @resize_up_down
    def get_arrayB(self):
        im1 = self.get_parentheses_set(type='composite_0')
        im2 = self.get_parentheses_set(type='composite_1')
        return im1, im2

    @resize_up_down
    def get_arrayC(self):
        im1 = self.get_parentheses_set(type='composite_0', dist_factor=5)
        im2 = self.get_parentheses_set(type='composite_1', dist_factor=5)
        return im1, im2

    @resize_up_down
    def get_arrayD(self):
        im1 = self.get_parentheses_crossed_set(type='composite_0', offset=0.2)
        im2 = self.get_parentheses_crossed_set(type='composite_1', offset=0.2)
        return im1, im2

    @resize_up_down
    def get_arrayE(self, dist_factor=10):
        def create_context():
            p = [self.img_size[0] / 2 - self.img_size[0] / dist_factor - self.img_size[0] * 0.2, self.img_size[1] / 2]
            self.draw_parentheses(p, 35, 'right', draw=draw)

            p = [self.img_size[0] / 2 + self.img_size[0] / dist_factor - self.img_size[0] * 0.15, self.img_size[1] / 2]
            self.draw_parentheses(p, 35, 'left', draw=draw)

        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        create_context()
        p = [self.img_size[0] / 2 + 1.35 * self.img_size[0] / dist_factor - self.img_size[0] * 0.067, self.img_size[1] / 2]
        self.draw_parentheses(p, 35, 'right', draw=draw)

        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        create_context()

        p = [self.img_size[0] / 2 + 1.35 * self.img_size[0] / dist_factor + self.img_size[0] * 0.067, self.img_size[1] / 2]
        self.draw_parentheses(p, 35, 'left', draw=draw)

        if self.antialiasing:
            im2 = self.apply_antialiasing(im2)

        return im1, im2

    def get_arrayF(self, dist_factor=7):
        im1, im2 = self.get_arrayE(dist_factor=dist_factor)
        return im1, im2

    @resize_up_down
    def get_arrayE2(self, dist_factor=10):
        """
        Like E, but when moving the middle parentesis the rest doesn't move
        """
        def create_context():
            p = [self.img_size[0] / 2 - self.img_size[0] / 10 - self.img_size[0] * 0.2, self.img_size[1] / 2]
            self.draw_parentheses(p, 35, 'right', draw=draw)

            p = [self.img_size[0] / 2 - self.img_size[0] / 10 + dist_factor, self.img_size[1] / 2]
            self.draw_parentheses(p, 35, 'left', draw=draw)

        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        create_context()
        p = [self.img_size[0] / 2 + 1.35 * self.img_size[0] / 6 - self.img_size[0] * 0.067, self.img_size[1] / 2]
        self.draw_parentheses(p, 35, 'right', draw=draw)

        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        create_context()

        p = [self.img_size[0] / 2 + 1.35 * self.img_size[0] / 10 + self.img_size[0] * 0.067, self.img_size[1] / 2]
        self.draw_parentheses(p, 35, 'left', draw=draw)

        if self.antialiasing:
            im2 = self.apply_antialiasing(im2)

        return im1, im2

    @resize_up_down
    def get_parentheses_crossed(self):
        im1 = self.get_parentheses_crossed_set(type='composite_0')
        im2 = self.get_parentheses_crossed_set(type='composite_1')
        return im1, im2

    def get_parentheses_crossed(self):
        im1 = self.get_parentheses_crossed_set(type='composite_0')
        im2 = self.get_parentheses_crossed_set(type='composite_1')
        return im1, im2

    @resize_up_down
    def get_circle_parentheses(self, par_dist_factor=30):
        # with dist_factor = 35 they are touching, the lower you go the farther away they get
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        circle_size_f = 4
        draw.arc([(im1.size[0]//2 - im1.size[0]//circle_size_f,
                   im1.size[0]//2 - im1.size[0]//circle_size_f),
                  (im1.size[0]//2 + im1.size[0] // circle_size_f, im1.size[0]//2 + im1.size[0] / circle_size_f)], 0, 360,  width=self.line_args['width'], fill=self.fill)
        im2 = self.get_parentheses_set(type='composite_1', dist_factor=par_dist_factor)
        return im1, im2

    @resize_up_down
    def get_square_rotarray2(self, rotate_segm=30):
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)
        s = 2.5  # increase this to reduce general size of the square
        ## top
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),

                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),

                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),

                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),

                                                (self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        scale_hole = 6  # inrease this to reduce hole in the square

        def rot_2d(theta):
            theta = np.deg2rad(theta)
            return np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        ## top left
        xy0 = np.array([self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                         self.img_size[1] // 2 - (self.img_size[0] // 2 // s)])
        xy1 = np.array([self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                         self.img_size[1] // 2 - (self.img_size[0] // scale_hole // s)])

        rot_vect = rot_2d(rotate_segm) @ (xy1 - xy0) + xy0
        self.draw_line_with_circled_edge(draw, (tuple(xy0), tuple(rot_vect)), width=self.line_args['width'])

        xy2 = np.array([self.img_size[0] // 2 - (self.img_size[0] // scale_hole // s),
                        self.img_size[1] // 2 - (self.img_size[0] // 2 // s)])

        rot_vect = rot_2d(rotate_segm) @ (xy2 - xy0) + xy0
        self.draw_line_with_circled_edge(draw, (tuple(xy0), tuple(rot_vect)), width=self.line_args['width'])


        ## top right
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // scale_hole // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),

                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // scale_hole // s))),
                                         width=self.line_args['width'])

        ## down right
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // scale_hole // s)),

                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // scale_hole // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),

                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])

        ## down left
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),

                                                (self.img_size[0] // 2 - (self.img_size[0] // scale_hole // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // scale_hole // s)),

                                                (self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        return im1, im2

    @resize_up_down
    def get_array2_rotarray2(self, rotate_segm=30):
        ## top
        im1, _ = self.get_array2()

        s = 2.5  # increase this to reduce general size of the square

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        scale_hole = 6  # inrease this to reduce hole in the square

        def rot_2d(theta):
            theta = np.deg2rad(theta)
            return np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])

        ## top left
        xy0 = np.array([self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                        self.img_size[1] // 2 - (self.img_size[0] // 2 // s)])
        xy1 = np.array([self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                        self.img_size[1] // 2 - (self.img_size[0] // scale_hole // s)])

        rot_vect = rot_2d(rotate_segm) @ (xy1 - xy0) + xy0
        self.draw_line_with_circled_edge(draw, (tuple(xy0), tuple(rot_vect)), width=self.line_args['width'])

        xy2 = np.array([self.img_size[0] // 2 - (self.img_size[0] // scale_hole // s),
                        self.img_size[1] // 2 - (self.img_size[0] // 2 // s)])

        rot_vect = rot_2d(rotate_segm) @ (xy2 - xy0) + xy0
        self.draw_line_with_circled_edge(draw, (tuple(xy0), tuple(rot_vect)), width=self.line_args['width'])

        ## top right
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // scale_hole // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),

                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 - (self.img_size[0] // scale_hole // s))),
                                         width=self.line_args['width'])

        ## down right
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // scale_hole // s)),

                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + (self.img_size[0] // scale_hole // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),

                                                (self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))),
                                         width=self.line_args['width'])

        ## down left
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),

                                                (self.img_size[0] // 2 - (self.img_size[0] // scale_hole // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // scale_hole // s)),

                                                (self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        return im1, im2

    @resize_up_down
    def get_array2_rotarray2_allseg(self, rotate_segm=30):
        ## top
        im1, _ = self.get_array2()

        s = 2.5  # increase this to reduce general size of the square

        im2 = self.create_canvas()
        draw = ImageDraw.Draw(im2)
        scale_hole = 6  # inrease this to reduce hole in the square

        def rot_2d(theta):
            theta = np.deg2rad(theta)
            return np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])

        ## top left
        xy0 = np.array([self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                        self.img_size[1] // 2 - (self.img_size[0] // 2 // s)])
        xy1 = np.array([self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                        self.img_size[1] // 2 - (self.img_size[0] // scale_hole // s)])

        rot_vect = rot_2d(rotate_segm) @ (xy1 - xy0) + xy0
        self.draw_line_with_circled_edge(draw, (tuple(xy0), tuple(rot_vect)), width=self.line_args['width'])

        xy2 = np.array([self.img_size[0] // 2 - (self.img_size[0] // scale_hole // s),
                        self.img_size[1] // 2 - (self.img_size[0] // 2 // s)])

        rot_vect = rot_2d(rotate_segm) @ (xy2 - xy0) + xy0
        self.draw_line_with_circled_edge(draw, (tuple(xy0), tuple(rot_vect)), width=self.line_args['width'])

        ## top right
        xy0 = np.array([self.img_size[0] // 2 + (self.img_size[0] // scale_hole // s),
                        self.img_size[1] // 2 - (self.img_size[0] // 2 // s)])
        xy1 = np.array([self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                       self.img_size[1] // 2 - (self.img_size[0] // 2 // s)])

        rot_vect = rot_2d(-rotate_segm)@(xy0 - xy1) + xy1
        self.draw_line_with_circled_edge(draw, (tuple(xy1), tuple(rot_vect)), width=self.line_args['width'])
        xy2 = np.array([self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                        self.img_size[1] // 2 - (self.img_size[0] // scale_hole // s)])

        rot_vect = rot_2d(-rotate_segm)@(xy2 - xy1) + xy1
        self.draw_line_with_circled_edge(draw, (tuple(xy1), tuple(rot_vect)), width=self.line_args['width'])


        ## down right
        xy0 = np.array([self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                        self.img_size[1] // 2 + (self.img_size[0] // scale_hole // s)])
        xy1= np.array([self.img_size[0] // 2 + (self.img_size[0] // 2 // s),
                       self.img_size[1] // 2 + (self.img_size[0] // 2 // s)])

        rot_vect = rot_2d(rotate_segm) @ (xy0 - xy1) + xy1
        self.draw_line_with_circled_edge(draw, (tuple(xy1), tuple(rot_vect)), width=self.line_args['width'])

        xy2 = np.array([self.img_size[0] // 2 + (self.img_size[0] // scale_hole // s),
                        self.img_size[1] // 2 + (self.img_size[0] // 2 // s)])

        rot_vect = rot_2d(rotate_segm) @ (xy2- xy1) + xy1
        self.draw_line_with_circled_edge(draw, (tuple(xy1), tuple(rot_vect)), width=self.line_args['width'])


        ## down left
        xy0 = np.array([self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                        self.img_size[1] // 2 + (self.img_size[0] // 2 // s)])
        xy1 = np.array([self.img_size[0] // 2 - (self.img_size[0] // scale_hole // s),
                        self.img_size[1] // 2 + (self.img_size[0] // 2 // s)])

        rot_vect = rot_2d(-rotate_segm) @ (xy1 - xy0) + xy0
        self.draw_line_with_circled_edge(draw, (tuple(xy0), tuple(rot_vect)), width=self.line_args['width'])

        xy2 = np.array([self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
                        self.img_size[1] // 2 + (self.img_size[0] // scale_hole // s)])

        rot_vect = rot_2d(-rotate_segm) @ (xy2 - xy0) + xy0
        self.draw_line_with_circled_edge(draw, (tuple(xy0), tuple(rot_vect)), width=self.line_args['width'])

        # self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
        #                                          self.img_size[1] // 2 + (self.img_size[0] // 2 // s)),
        #
        #                                         (self.img_size[0] // 2 - (self.img_size[0] // scale_hole // s),
        #                                          self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))
        #
        # self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
        #                                          self.img_size[1] // 2 + (self.img_size[0] // scale_hole // s)),
        #
        #                                         (self.img_size[0] // 2 - (self.img_size[0] // 2 // s),
        #                                          self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        return im1, im2


    # @resize_up_down
    def get_rotate_triangle(self):
        im1 = self.get_triangle(type='composite_0')
        im2 = im1.rotate(90)

        return im1, im2

    @resize_up_down
    def get_curly_composite_with_space(self, space=0):
        # im1 = self.get_char_canvas_with_rot('{' + " ".join(['']*(space+1)) + '(', sz_fact=sz, font_size=240, font_path='./AmasisMTpro.ttf')
        # im2 = self.get_char_canvas_with_rot('{' + " ".join(['']*(space+1)) + ')', sz_fact=sz, font_size=240, font_path='./AmasisMTpro.ttf')
        #
##

        def add_curly_bracket(self, im):
            font_path = 'C:/windows/Fonts/calibri.ttf'
            rot = 0
            font_size = 265
            cc = self.create_canvas(img_size=sz_fact, borders=False)
            draw = ImageDraw.Draw(cc)
            arial = ImageFont.truetype(font_path, font_size)
            draw.text((sz_fact[0] / 2, sz_fact[1] / 2), '{', font=arial, anchor="mm", fill=(self.line_col,)*3)
            cc = cc.rotate(rot, resample=2, center=(sz_fact[0] // 2, sz_fact[1] // 2))

            im.paste(cc, (self.img_size[0] // 2 - sz_fact[0] // 2 - 30,
                       self.img_size[1] // 2 - sz_fact[1] // 2 + 5,
                       self.img_size[0] // 2 - sz_fact[0] // 2 + sz_fact[0] - 30,
                       self.img_size[1] // 2 - sz_fact[1] // 2 + sz_fact[1] + 5))

        sz_fact = (110, 240)
        im1 = self.create_canvas()
        drawIm = ImageDraw.Draw(im1)

        add_curly_bracket(self, im1)
        dist_factor = 10
        p = [self.img_size[0] / 2 + self.img_size[0] / dist_factor - self.img_size[0] * 0.067, self.img_size[1] / 2]
        self.draw_parentheses(p, 35, 'right', draw=drawIm)

        sz_fact = (110, 240)
        im2 = self.create_canvas()
        drawIm = ImageDraw.Draw(im2)
        add_curly_bracket(self, im2)
        dist_factor = 10
        p = [self.img_size[0] / 2 + self.img_size[0] / dist_factor + self.img_size[0] * 0.09, self.img_size[1] / 2]
        self.draw_parentheses(p, 35, 'left', draw=drawIm)

        return im1, im2

    @resize_up_down
    def get_curly_base(self):
        sz = (450, 450)
        im1 = self.get_char_canvas_with_rot('{', sz_fact=sz, font_size=240)
        im2 = self.get_char_canvas_with_rot('}', sz_fact=sz, font_size=240)
        return im1, im2

    @resize_up_down
    def get_brackets_base(self):
        sz = (450, 450)
        im1 = self.get_char_canvas_with_rot('(', sz_fact=sz, font_size=240, font_path='./AmasisMTpro.ttf')
        im2 = self.get_char_canvas_with_rot(')', sz_fact=sz, font_size=240, font_path='./AmasisMTpro.ttf')
        return im1, im2


    @resize_up_down
    def get_brackets_composite_with_space(self, space=0):
        sz = (450, 450)
        im1 = self.get_char_canvas_with_rot('(' + " ".join(['']*(space+1)) + ')', sz_fact=sz, font_size=240, font_path='./AmasisMTpro.ttf')
        im2 = self.get_char_canvas_with_rot('(' + " ".join(['']*(space+1)) + '(', sz_fact=sz, font_size=240, font_path='./AmasisMTpro.ttf')
        return im1, im2


    @resize_up_down
    def get_array11_curly(self):
        ##
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)

        s = 2.5

        char = '{'
        sz_fact = (400, 400)
        cc = self.create_canvas(img_size=sz_fact, borders=False)
        draw = ImageDraw.Draw(cc)

        arial = ImageFont.truetype("C:/windows/Fonts/inkfree.ttf", 190)
        draw.text((sz_fact[0] / 2, sz_fact[1] / 2), char, font=arial, anchor="mm",  fill=(self.line_col,)*3, stroke_width=5)
        cc = cc.rotate(90, resample=2, center=(sz_fact[0] // 2, sz_fact[1] // 2))
        im1.paste(cc, (self.img_size[0] // 2 - sz_fact[0] // 2 -5,
                       self.img_size[1] // 2 - sz_fact[1] // 2 + 110,
                       self.img_size[0] // 2 - sz_fact[0] // 2 + sz_fact[0] -5,
                       self.img_size[1] // 2 - sz_fact[1] // 2 + sz_fact[1] + 110))
        draw = ImageDraw.Draw(im1)

        ###
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s,
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 - self.img_size[0] // 2 // s,
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s-8))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s,
                                                 self.img_size[0] // 2 + self.img_size[0] // 2 // s),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s,
                                                 self.img_size[0] // 2 - self.img_size[0] // 2 // s)))

        im2 = self.create_canvas()
        char = '{'
        sz_fact = (400, 400)
        cc = self.create_canvas(img_size=sz_fact, borders=False)
        draw = ImageDraw.Draw(cc)

        arial = ImageFont.truetype("C:/windows/Fonts/inkfree.ttf", 190)
        draw.text((sz_fact[0] / 2, sz_fact[1] / 2), char, font=arial, anchor="mm", fill=(self.line_col,)*3, stroke_width=5)
        cc = cc.rotate(90, resample=2, center=(sz_fact[0] // 2, sz_fact[1] // 2))
        im2.paste(cc, (self.img_size[0] // 2 - sz_fact[0] // 2 - 5 ,
                       self.img_size[1] // 2 - sz_fact[1] // 2 + 110,
                       self.img_size[0] // 2 - sz_fact[0] // 2 + sz_fact[0] - 5,
                       self.img_size[1] // 2 - sz_fact[1] // 2 + sz_fact[1] + 110))
        draw = ImageDraw.Draw(im2)

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s,
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                                (self.img_size[0] // 2 - self.img_size[0] // 2 // s,
                                                 self.img_size[1] // 2 + (self.img_size[0] // 2 // s - 8))))
        # self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s, self.img_size[0] // 2 + self.img_size[0] // 2 // s),
        #                                         (self.img_size[0] // 2 + self.img_size[0] // 2 // s, self.img_size[0] // 2 + self.img_size[0] // 2 // s)))
        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s,
                                                 self.img_size[0] // 2 - self.img_size[0] // 2 // s),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s - 20,
                                                 self.img_size[0] // 2 + self.img_size[0] // 2 // s - 8)))

        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

    @resize_up_down
    def get_array4_curly(self):
        im1 = self.create_canvas()
        draw = ImageDraw.Draw(im1)

        s = 2.5

        char = '{'
        sz_fact = (400, 400)
        cc = self.create_canvas(img_size=sz_fact, borders=False)
        draw = ImageDraw.Draw(cc)

        arial = ImageFont.truetype("C:/windows/Fonts/inkfree.ttf", 190)
        draw.text((sz_fact[0] / 2, sz_fact[1] / 2), char, font=arial, anchor="mm",  fill=(self.line_col,)*3, stroke_width=5)
        cc = cc.rotate(90, resample=2, center=(sz_fact[0] // 2, sz_fact[1] // 2))
        im1.paste(cc, (self.img_size[0] // 2 - sz_fact[0] // 2 -5,
                       self.img_size[1] // 2 - sz_fact[1] // 2 + 115,
                       self.img_size[0] // 2 - sz_fact[0] // 2 + sz_fact[0] -5,
                       self.img_size[1] // 2 - sz_fact[1] // 2 + sz_fact[1] + 115))
        draw = ImageDraw.Draw(im1)

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s,
                                                 self.img_size[1] // 2 - (self.img_size[0] // 2 // s)),
                                               (self.img_size[0] // 2 - self.img_size[0] // 2 // s,
                                                self.img_size[1] // 2 + (self.img_size[0] // 2 // s))))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s ,
                                                 self.img_size[0] // 2 - self.img_size[0] // 2 // s),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s - 20,
                                                 self.img_size[0] // 2 + self.img_size[0] // 2 // s - 8 )))
        im2 = self.create_canvas()
        char = '{'
        sz_fact = (400, 400)
        cc = self.create_canvas(img_size=sz_fact, borders=False)
        draw = ImageDraw.Draw(cc)

        arial = ImageFont.truetype("C:/windows/Fonts/inkfree.ttf", 190)
        draw.text((sz_fact[0] / 2, sz_fact[1] / 2), char, font=arial, anchor="mm", fill=(self.line_col,)*3, stroke_width=5)
        cc = cc.rotate(90, resample=2, center=(sz_fact[0] // 2, sz_fact[1] // 2))
        im2.paste(cc, (self.img_size[0] // 2 - sz_fact[0] // 2,
                       self.img_size[1] // 2 - sz_fact[1] // 2 + 110,
                       self.img_size[0] // 2 - sz_fact[0] // 2 + sz_fact[0],
                       self.img_size[1] // 2 - sz_fact[1] // 2 + sz_fact[1] + 110))
        draw = ImageDraw.Draw(im2)

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 + self.img_size[0] // 2 // s - 8,
                                                 self.img_size[0] // 2 + self.img_size[0] // 2 // s),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s - 8, self.img_size[0] // 2 - self.img_size[0] // 2 // s)))

        self.draw_line_with_circled_edge(draw, ((self.img_size[0] // 2 - self.img_size[0] // 2 // s ,
                                                 self.img_size[0] // 2 - self.img_size[0] // 2 // s),
                                                (self.img_size[0] // 2 + self.img_size[0] // 2 // s - 8, self.img_size[0] // 2 + self.img_size[0] // 2 // s)))
        if self.antialiasing:
            im1 = self.apply_antialiasing(im1)
            im2 = self.apply_antialiasing(im2)
        return im1, im2

##

# dr = DrawShape(background='random', img_size=(224, 224), width=10, min_dist_bw_points=20, min_dist_borders=40)# im = dr.get_array2_rotarray2_allseg(rotate_segm=315)
# # # im = dr.get_array11_curly()
# # im = dr.get_array2_rotarray2_allseg(space=5)
# im = dr.get_curly_composite_with_space(space=0)
# # im = dr.get_brackets_base()
#
# im[0].show()
# im[1].show()
# #

##

