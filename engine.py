import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt 
import datetime
from PIL import Image


## define a new class:

def dist2d(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class Point():
    """
    A class representing a point in 3D space.

    Attributes:
    - x: float, the x-coordinate of the point
    - y: float, the y-coordinate of the point
    - z: float, the z-coordinate of the point

    Methods:
    - numpy(): returns the point as a numpy array
    - dist(p): returns the Euclidean distance between this point and another point p
    - cross(p): returns the cross product of this point and another point p
    - __repr__(): returns a string representation of the point
    - __sub__(p): returns the vector difference between this point and another point p
    - __mul__(p): returns the dot product of this point and another point p
    - normalize(): returns the normalized version of this point
    - norm(): returns the Euclidean norm of this point
    - angle2d(direction): returns the 2D angle between this point and a given direction vector
    - get_2d_coords(pov, direction, vision_angle, resolution_x, resolution_y, verbose): returns the 2D coordinates of this point in an image
    """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def numpy(self):
        return np.asarray([self.x,self.y,self.z])
        
    def dist(self, p):
        return np.sqrt((self.x - p.x)**2 + (self.y - p.y)**2 + (self.z - p.z)**2)

    def cross(self,p):
        return Point(self.y*p.z - self.z*p.y, self.z*p.x - self.x*p.z, self.x*p.y - self.y*p.x)
    
    def __repr__(self):
        return f'({self.x}, {self.y}, {self.z})'

    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y, self.z - p.z)

    def __mul__(self, p):
        return self.x * p.x + self.y * p.y + self.z * p.z
    
    def normalize(self):
        mnorm =  self.norm()
        if mnorm == 0.0:
            return Point(0.0, 0.0, 0.0)
        return Point(self.x/mnorm, self.y/mnorm, self.z/mnorm)

    def norm(self):
        norm = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        return norm
    
    def angle2d(self, direction):
        """
        Returns the 2D angle between this point and a given direction vector.

        Parameters:
        - direction: Point, the direction vector

        Returns:
        - theta: float, the horizontal angle between this point and the direction vector
        - phi: float, the vertical angle between this point and the direction vector
        """
        p_xy = Point(self.x, self.y, 0)

        theta = np.arctan2(self.x,self.y) - np.arctan2(direction.x,direction.y)
        phi = np.arccos(self.z / self.norm()) - np.arccos(direction.z)
        return theta, phi

    def get_2d_coords(self, pov, direction, vision_angle, resolution_x, resolution_y, verbose):
        """
        Returns the 2D coordinates of this point in an image.

        Parameters:
        - pov: Point, the point of view
        - direction: Point, the direction vector
        - vision_angle: float, the field of view angle
        - resolution_x: int, the horizontal resolution of the image
        - resolution_y: int, the vertical resolution of the image
        - verbose: int, the verbosity level

        Returns:
        - x: int, the horizontal pixel coordinate of this point in the image
        - y: int, the vertical pixel coordinate of this point in the image
        """
        v = self - pov
        theta, phi = v.angle2d(direction)

        vision_angle = np.tan(vision_angle)
        phi = np.tan(phi)/ (np.abs(np.cos(theta)))
        theta = np.tan(theta)
        
        x = int((resolution_x//2) * (theta / vision_angle)) + resolution_x//2
        y = int((resolution_y//2) * (phi / vision_angle)) + resolution_y//2

        if verbose == 3:
            print(v)
            print((theta / vision_angle,phi / vision_angle))
            print((x,y))
        return x, y

class Quad():
    """
    A class representing a quadrilateral in 3D space.

    Attributes:
    - p1: Point, the first vertex of the quadrilateral
    - p2: Point, the second vertex of the quadrilateral
    - p3: Point, the third vertex of the quadrilateral
    - p4: Point, the fourth vertex of the quadrilateral
    - center: Point, the center point of the quadrilateral
    - id: int, the ID of the quadrilateral
    - color: str, the color of the quadrilateral
    - desc: str, the description of the quadrilateral
    - boxid: int, the ID of the bounding box
    - res: float, the resolution of the quadrilateral

    Methods:
    - __repr__(): returns a string representation of the quadrilateral
    - mindist(p): returns the minimum distance between this quadrilateral and a given point p
    - draw_quad(image, mapping, normal_map, depth_map , pov, direction, vision_angle, intensity, verbose, forcecolor, linewidth): draws the quadrilateral on an image
    """

    counter = 1

    def __init__(self, p1, p2, p3, p4 , color = 'red', desc = 'plain', boxid = None, res = 1):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.center = Point((p1.x + p2.x + p3.x + p4.x)/4, (p1.y + p2.y + p3.y + p4.y)/4, (p1.z + p2.z + p3.z + p4.z)/4)
        self.id = Quad.counter
        self.color = color
        self.desc = desc
        self.boxid = boxid
        
        Quad.counter += 1

        if res is None:
            res = 64.0    
            while (int(np.ceil(p1.dist(p2)/res)) <= 512 and int(np.ceil(p1.dist(p4)/res)) <= 512):
                res = res/2
        
        self.res = res
        
        self.shape = (int(np.ceil(p1.dist(p2)/res)), int(np.ceil(p1.dist(p4)/res)))

        while (self.shape[0] > 1024 or self.shape[1] > 1024):
            res = res*2
            self.shape = (int(np.ceil(p1.dist(p2)/res)), int(np.ceil(p1.dist(p4)/res)))

        assert(self.shape[0] <= 1024)
        assert(self.shape[1] <= 1024)
        
        self.img = np.zeros((self.shape[0], self.shape[1],3), dtype = np.uint8 )

        self.img_src_count = np.zeros((self.shape[0], self.shape[1]), dtype = np.uint8 )
        
        v1 = (p2 - p1)
        v2 = (p4 - p1)
        
        v_norm = v1.cross(v2).normalize()
        self.v_norm = v_norm

    def __repr__(self):
        return f'Quad({self.p1}, {self.p2}, {self.p3}, {self.p4})'

    def mindist(self,p):
        """
        Returns the minimum distance between this quadrilateral and a given point p.

        Parameters:
        - p: Point, the point to calculate the distance to

        Returns:
        - dist: float, the minimum distance between this quadrilateral and the point p
        """
        d1 = p.dist(self.p1)
        d2 = p.dist(self.p2)
        d3 = p.dist(self.p3)
        d4 = p.dist(self.p4)
        dc = p.dist(self.center)
        return np.min([d1,d2,d3,d4])
    
    def draw_quad(self, image, mapping, normal_map, depth_map , pov, direction, vision_angle, intensity, verbose, forcecolor, linewidth = 0):
        """
        Draws the quadrilateral on an image.

        Parameters:
        - image: numpy array, the image to draw on
        - mapping: numpy array, the mapping of the image to 3D space
        - normal_map: numpy array, the normal map of the image
        - depth_map: numpy array, the depth map of the image
        - pov: Point, the point of view
        - direction: Point, the direction vector
        - vision_angle: float, the field of view angle
        - intensity: float, the intensity of the quadrilateral
        - verbose: int, the verbosity level
        - forcecolor: str, the color to use for the quadrilateral
        - linewidth: int, the width of the line to draw

        Returns:
        - image: numpy array, the updated image
        """
        resolution_x = image.shape[0]
        resolution_y = image.shape[1]

        ## get the 2d coordinates of the 4 points
        p1 = self.p1.get_2d_coords(pov, direction, vision_angle, resolution_x, resolution_y, verbose)
        p2 = self.p2.get_2d_coords(pov, direction, vision_angle, resolution_x, resolution_y, verbose)
        p3 = self.p3.get_2d_coords(pov, direction, vision_angle, resolution_x, resolution_y, verbose)
        p4 = self.p4.get_2d_coords(pov, direction, vision_angle, resolution_x, resolution_y, verbose)
        
        if not  image is None:
            if forcecolor is None:
                color = self.color
            else:
                color = forcecolor

            ## draw the quad
            line_aa = 0
            ##line_aa = cv2.LINE_AA        
            pp = np.array([p1, p2, p3, p4])
            image = cv2.drawContours(image, [pp], 0, color, -1, line_aa)

            if verbose == 1:
                plt.imshow(image)
                plt.show()

            if linewidth > 0:
                image = cv2.line(image, p1, p2, (0,0,0), linewidth)
                image = cv2.line(image, p2, p3, (0,0,0), linewidth)
                image = cv2.line(image, p3, p4, (0,0,0), linewidth)
                image = cv2.line(image, p4, p1, (0,0,0), linewidth)
        
        pts1 = np.float32([[0,0],
                        [0,self.shape[0]],
                        [self.shape[1],self.shape[0]],
                        [self.shape[1],0]])


        pts2 = np.float32([p1 , p2, p3, p4])
        transformation_matrix = cv2.getPerspectiveTransform(pts1,pts2, cv2.DECOMP_SVD )

        if not mapping is None:
            x = np.arange(0,self.shape[0])
            y = np.arange(0,self.shape[1])
            yy, xx = np.meshgrid(y,x)




            ptr_img = np.zeros((self.shape[0], self.shape[1],3), dtype = np.uint8)

            ptr_img[:,:,0] = self.id % 256
            ptr_img[:,:,1] = np.round(xx) % 256
            ptr_img[:,:,2] = np.round(yy) % 256

            new_mapping1 = cv2.warpPerspective(ptr_img, transformation_matrix, (resolution_x, resolution_y))

            ptr_img2 = np.zeros((self.shape[0], self.shape[1],3), dtype = np.uint8)
            ptr_img2[:,:,0] = self.id // 256
            ptr_img2[:,:,1] = np.round(xx) // 256
            ptr_img2[:,:,2] = np.round(yy) // 256

            new_mapping2 = cv2.warpPerspective(ptr_img2, transformation_matrix, (resolution_x, resolution_y))

            ## image only takes non zero values from tar
            mapping = np.where(np.expand_dims(new_mapping1[:,:,0] == 0,2), mapping, new_mapping1 + new_mapping2*256)

        
        if not normal_map is None:
            dv = (self.v_norm - direction).normalize().numpy()   
            dv = np.round((dv+1)*128)
            
            normal_map = cv2.drawContours(normal_map, [pp], 0, dv , -1, line_aa)

            if verbose == 5:
                ##import pdb; pdb.set_trace()
                plt.imshow(normal_map)
                plt.show()
        
        if not depth_map is None:
            d1 = pov.dist(self.p1)

            x = np.arange(0,self.shape[0])
            y = np.arange(0,self.shape[1])
            yy, xx = np.meshgrid(y,x)
            
            cord_x = self.p1.x + xx*(self.p2.x - self.p1.x)/self.shape[0] + yy*(self.p4.x - self.p1.x)/self.shape[1]
            cord_y = self.p1.y + xx*(self.p2.y - self.p1.y)/self.shape[0] + yy*(self.p4.y - self.p1.y)/self.shape[1]
            cord_z = self.p1.z + xx*(self.p2.z - self.p1.z)/self.shape[0] + yy*(self.p4.z - self.p1.z)/self.shape[1]

            dist_map_origin = np.sqrt((cord_x - pov.x)**2 + (cord_y - pov.y)**2 + (cord_z - pov.z)**2)
            
            new_depth = cv2.warpPerspective(dist_map_origin, transformation_matrix, (resolution_x, resolution_y))

            ##min_depth_map = np.where(new_depth < depth_map, new_depth, depth_map)
            depth_map = np.where(new_depth == 0 , depth_map, new_depth )

            

            if verbose == 6:
                ##import pdb; pdb.set_trace()
                plt.imshow(depth_map)
                plt.show()
            

        
        return image, mapping, normal_map, depth_map

    def unstamp(self, image, pov, direction, vision_angle, intensity, verbose):
        resolution_x = image.shape[0]
        resolution_y = image.shape[1]

        ## get the 2d coordinates of the 4 points
        p1 = self.p1.get_2d_coords(pov, direction, vision_angle, resolution_x, resolution_y, verbose)
        p2 = self.p2.get_2d_coords(pov, direction, vision_angle, resolution_x, resolution_y, verbose)
        p3 = self.p3.get_2d_coords(pov, direction, vision_angle, resolution_x, resolution_y, verbose)
        p4 = self.p4.get_2d_coords(pov, direction, vision_angle, resolution_x, resolution_y, verbose)
        
        line_aa = 0
        ##line_aa = cv2.LINE_AA        
        
        if verbose == 1:
            plt.imshow(image)
            plt.show()
        
        pts1 = np.float32([[0,0],
                        [0,self.shape[0]],
                        [self.shape[1],self.shape[0]],
                        [self.shape[1],0]])

        pts2 = np.float32([p1 , p2, p3, p4])

        transformation_matrix = cv2.getPerspectiveTransform(pts1,pts2, cv2.DECOMP_SVD)
        
        unstamped_img = cv2.warpPerspective(self.img, transformation_matrix, (resolution_x, resolution_y))

        ## image only takes non zero values from tar
        image = np.where(np.expand_dims(unstamped_img[:,:,0] == 0,2), image, unstamped_img)

        return image


class Box():
    
    counter = 1

    
    def __init__(self, p, w, h, d, color = 'red', desc = 'box', res = None):
        self.x = p.x
        self.y = p.y
        self.z = p.z
        self.p = Point(self.x + w/2, self.y + h/2, self.z+h/2)
        self.w = w 
        self.h = h
        self.d = d
        self.color = color
        self.desc = desc
        self.id = Box.counter
        self.res = res
        
        self.p000, self.p100, self.p110, self.p010, self.p001, self.p101, self.p111, self.p011 = self.get_points()
        self.quads = self.get_quads()

        Box.counter += 1
        
    def get_points(self):
        points = [  Point(self.x, self.y, self.z),
                    Point(self.x + self.w, self.y, self.z),
                    Point(self.x + self.w, self.y + self.h, self.z),
                    Point(self.x, self.y + self.h, self.z),
                    Point(self.x, self.y, self.z + self.d),
                    Point(self.x + self.w, self.y, self.z + self.d),
                    Point(self.x + self.w, self.y + self.h, self.z + self.d),
                    Point(self.x, self.y + self.h, self.z + self.d)
                 ]
        return points

    def get_quads(self):
        quad_x0 = Quad(self.p000, self.p001, self.p011, self.p010, self.color, self.desc, self.id, res = self.res) ## x= 0
        quad_y0 = Quad(self.p000, self.p100, self.p101, self.p001, self.color, self.desc, self.id, res = self.res) ## y= 0
        quad_z0 = Quad(self.p000, self.p010, self.p110, self.p100, self.color, self.desc, self.id, res = self.res) ## z= 0
        quad_x1 = Quad(self.p100, self.p110, self.p111, self.p101, self.color, self.desc, self.id, res = self.res) ## x= 1
        quad_y1 = Quad(self.p010, self.p110, self.p111, self.p011, self.color, self.desc, self.id, res = self.res) ## y= 1
        quad_z1 = Quad(self.p001, self.p101, self.p111, self.p011, self.color, self.desc, self.id, res = self.res) ## z= 1

        # quad_x1 = Quad(self.p100, self.p101, self.p111, self.p110, self.color, self.desc, self.id, res = self.res) ## x= 1
        # quad_y1 = Quad(self.p010, self.p011, self.p111, self.p110, self.color, self.desc, self.id, res = self.res) ## y= 1
        # quad_z1 = Quad(self.p001, self.p011, self.p111, self.p101, self.color, self.desc, self.id, res = self.res) ## z= 1

        return [quad_x0, quad_y0, quad_z0, quad_x1, quad_y1, quad_z1]


        

class Sketch3d():    
    def __init__(self):

        ## sortable list
        
        self.boxes = []
        self.flats = []


    def add_box(self, box):
        self.boxes.append(box)

    def add_flat(self, quad):
        self.flats.append(quad)

    def get_quads(self, pov, direction):        
        quads = []
        for box in self.boxes:
            quads += box.quads

        for flat in self.flats:
            quads.append(flat)

        ##quads = [quad for quad in quads if (quad.center- pov) * direction > 0]

        sort_quad = sorted(quads, key=lambda quad: pov.dist(quad.center), reverse=True)



        return sort_quad



class Scene():
    def __init__(self, sketch3d, pov, direction , vision_angle = 30, resolution = (1024,1024), intensity = 0.7, verbose = 0, linewidth = 0.0, use_normal_map = True, use_depth_map = True):

        
        
        vision_angle = np.pi*vision_angle/180
        direction = direction.normalize()

        seg_map = np.ones((resolution[0], resolution[1],3), dtype = np.uint8 )*255
        
        if use_normal_map:
            normal_map = np.zeros((resolution[0], resolution[1],3), dtype = np.uint8 )
        else:
            normal_map = None
            
        if use_depth_map:
            depth_map = np.zeros((resolution[0], resolution[1]), dtype = np.float32 )
        else:
            depth_map = None
            
        mapping = np.zeros((resolution[0], resolution[1],3), dtype = np.int)

        self.sketch = sketch3d
        self.quads = sketch3d.get_quads(pov, direction)
        self.resolution = resolution
        self.pov = pov
        self.vision_angle = vision_angle
        self.direction = direction
        
        
        color = None
        for quad in self.quads:
            seg_map, mapping, normal_map, depth_map = quad.draw_quad(seg_map, mapping, normal_map, depth_map , pov, direction, vision_angle, intensity = intensity, verbose = verbose, forcecolor = color, linewidth = linewidth)
            if verbose == 2:
                seg_map_presentation = np.ones((resolution[0], resolution[1],3), dtype = np.uint8 )*255
                mapping_misc = np.zeros((resolution[0], resolution[1],3), dtype = np.uint8)
                normal_map_presentation = np.zeros((resolution[0], resolution[1],3), dtype = np.uint8 )
                depth_map_presentation = np.zeros((resolution[0], resolution[1],3), dtype = np.uint8 )

                seg_map_presentation, _, normal_map_presentation, depth_map_presentation = quad.draw_quad(seg_map_presentation, mapping_misc, normal_map_presentation, depth_map_presentation , pov, direction, vision_angle, intensity = intensity, verbose = verbose, forcecolor = color)
                plt.imshow(seg_map_presentation)
                plt.show()
                plt.imshow(normal_map_presentation)
                plt.show()
                plt.imshow(depth_map_presentation)
                plt.show()
                print(quad.id)


        self.seg_map = seg_map
        self.mapping = mapping



        ##initialize the image to be random noise:

        ##self.image = np.random.randint(0,255,(resolution[0], resolution[1],3), dtype = np.uint8 )
        self.image = np.zeros((resolution[0], resolution[1],3), dtype = np.uint8 )


        # plt.imshow(image)
        # plt.show()
        
        # plt.imshow(mapping)
        # plt.show()
        
        if use_normal_map:
            self.normal_map = normal_map
            # plt.imshow(normal_map)
            # plt.show()
        
        if use_depth_map:
            self.initialize_depth(depth_map)
            # plt.imshow(self.depth_map)
            # plt.show()
        
    def stamp(self, image, verbose = False):
        ## get unique ids from mapping:
        ids = np.unique(self.mapping[:,:,0])

        image_cp = image.copy()
        for id in ids:
            if id == 0:
                continue

            for quad in self.quads:
                if quad.id == id:
                    break
                    
            if not quad.id == id:
                ##print("skipping ",id)
                continue
            quad.img[:,:,:] = 0
            idx1, idx2 = np.where(self.mapping[:,:,0] == id)
            
            for i in range(len(idx1)):
                ix1 = idx1[i]
                ix2 = idx2[i]
                
                color = image[ix1,ix2,:]
                qid, pos_x, pos_y = self.mapping[ix1,ix2,:]
                
                valid = True
                
                if pos_x >= quad.img.shape[0]:
                    image_cp[ix1,ix2,0] = 255
                    valid = False
                if pos_y >= quad.img.shape[1]:
                    image_cp[ix1,ix2,2] = 255
                    valid = False
                if pos_x < 0:
                    image_cp[ix1,ix2,0] = 0
                    valid = False
                if pos_y < 0:
                    image_cp[ix1,ix2,2] = 0
                    valid = False
                            
                if valid:
                    if np.all(quad.img[pos_x,pos_y,:] == (0,0,0)):
                        image_cp[ix1,ix2,1] = 255
                    
                    quad.img[pos_x,pos_y,:] = color
            
            if verbose:
                plt.imshow(quad.img)
                plt.show()
                plt.imshow(image_cp)
                plt.show()
        

                
                
    def unstamp(self, intensity = 1.0, verbose = False):
        vision_angle = self.vision_angle
        direction = self.direction

        pov = self.pov
        resolution = self.resolution
        
        self.quads = self.sketch.get_quads(pov, direction)
        
        image = np.zeros((resolution[0], resolution[1],3), dtype = np.uint8 )

        color = None
        for quad in self.quads:
            image = quad.unstamp(image, pov, direction, vision_angle, intensity = intensity, verbose = verbose)
            
        if verbose:
            plt.imshow(image)
        
        self.update_image(image, intensity)

        return image
    
    def update_image(self, new_image, intensity= 1.0):
        merged_img = self.image * (1 - intensity) + new_image * intensity
        merged_img = np.round(merged_img).astype(np.uint8)
        mask = np.expand_dims(((new_image == 0).sum(2) == 3),2)
        self.set_image(np.where(mask, self.image, merged_img))
    
    def initialize_depth(self, depth_map):
        depth_map = np.float32(depth_map)
        image_depth = depth_map.copy()
        ##depth_map = depth_map - np.min(depth_map[depth_map>0])
        image_depth = np.where( image_depth == 0, 0, np.max(image_depth) - image_depth)
        image_depth = image_depth/np.max(image_depth)

        bg_threhold = 0.4
        
        ##import pdb; pdb.set_trace()

        x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < bg_threhold] = 0

        y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < bg_threhold] = 0

        z = np.ones_like(x) * np.pi * 2.0

        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    
        self.depth_map = image
    
    def get_seg_map(self):
        return self.seg_map
    
    def get_norm_map(self):
        return self.normal_map
    
    def get_depth_map(self):
        return self.depth_map

    def get_image(self):
        return self.image
    
    def set_image(self, image):
        self.image = image

    def transition(self, scenes, intensity = 1.0):
        if isinstance(scenes, Scene):
            scenes = [scenes]

        stamped_img = self.get_image()
        self.stamp(stamped_img, verbose = False)

        for original_scene in scenes:
            unstamped_img = original_scene.unstamp(intensity = intensity, verbose = False)


def color_dist(a,b):
    return np.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

def random_color(r=None,g=None,b=None):
    r = r if r is not None else np.round(32*np.random.uniform())*8
    g = g if g is not None else np.round(32*np.random.uniform())*8
    b = b if b is not None else np.round(32*np.random.uniform())*8
        
#     while color_dist((r,g,b),(200,200,255))< (110):
#         r = 255*np.random.uniform()
#         g = 255*np.random.uniform()
#         b = 255*np.random.uniform()
        
    return  (r,g,b)


def gen_farm():
    skretch = Sketch3d()

    skretch.add_box(Box(Point(20,60,0), 15, 15, 30, color = random_color(0,0,255)))

    skretch.add_box(Box(Point(100,130,0), 15, 15, 100, color = random_color(255,0,0)))

    skretch.add_box(Box(Point(60,20,0), 70, 50, 20, color = random_color(100,100,100), res = 0.5))

    skretch.add_box(Box(Point(10,10,0), 30, 30, 1, color = (0,255,0)))

    return skretch

def add_skies(skretch, segments=1, Z = 300,color = (175,175,255)):

    size = 2**14//segments
    for i in range(segments):
        for j in range(segments):
            flat = Quad(Point(-10+size*i,-10+size*j,Z), 
                        Point(-10+size*(i+1),-10+size*j,Z), 
                        Point(-10+size*(i+1),-10+size*(j+1),Z), 
                        Point(-10+size*i,-10+size*(j+1),Z), 
                        color = color)

            skretch.add_flat(flat)

    return skretch




def gen_metropolis(num_of_background_buildings= 100):
    """
    Generates a 3D sketch of a city with random buildings and returns it.

    Args:
    - num_of_background_buildings (int): The number of random buildings to generate in the background. Default is 100.

    Returns:
    - skretch (Sketch3d): A 3D sketch of the city with the generated buildings.
    """
    skretch = Sketch3d()

    skretch.add_box(Box(Point(60,90,0), 15, 15, 30, color = random_color(None,None,None)))
    skretch.add_box(Box(Point(40,90,0), 15, 15, 30, color = random_color(None,None,None)))
    skretch.add_box(Box(Point(20,90,0), 15, 15, 30, color = random_color(None,None,None)))
    skretch.add_box(Box(Point(0,90,0), 15, 15, 30, color = random_color(None,None,None)))


    skretch.add_box(Box(Point(60,130,0), 15, 15, 30, color = random_color(None,None,None)))
    skretch.add_box(Box(Point(40,130,0), 15, 15, 30, color = random_color(None,None,None)))
    skretch.add_box(Box(Point(20,130,0), 15, 15, 30, color = random_color(None,None,None)))
    skretch.add_box(Box(Point(0,130,0), 15, 15, 30, color = random_color(None,None,None)))

    skretch.add_box(Box(Point(70,60,0), 40, 20, 20, color = random_color(None,None,None)))
    skretch.add_box(Box(Point(70,30,0), 40, 20, 20, color = random_color(None,None,None)))
    skretch.add_box(Box(Point(70,0,0), 40, 20, 20, color = random_color(None,None,None)))

    # skretch.add_box(Box(Point(100,140,0), 20, 10, 50, color = (255,255,0)))
    # skretch.add_box(Box(Point(240,200,0), 20, 20, 100, color = (255,255,0)))
    # skretch.add_box(Box(Point(50,1000,0), 20, 20, 150, color = (255,255,0)))


    skretch.add_box(Box(Point(10,10,0), 50, 50, 1, color = (0,255,0)))
    skretch.add_box(Box(Point(20,20,0), 30, 30, 1, color = random_color(None,None,None)))

    ##skretch.add_box(Box(Point(-100,-100,1000), 10000, 10000, 0, color = (175,175,255)))
    ##skretch.add_box(Box(Point(1,1,0), 10000, 10000, 0, color = (0,0,0)))


    for _ in range(num_of_background_buildings):
        
        x = np.random.uniform()*1000
        y = np.random.uniform()*1000
        
        while x < 100 and y< 100:
            x = np.random.uniform()*1000
            y = np.random.uniform()*1000
        h = np.random.uniform()*30 + 10
        w = np.random.uniform()*30 + 10
        d = np.random.uniform()*200 + 100
        
        skretch.add_box(Box(Point(x,y,0), h, w, d, color = random_color() ))

    return skretch

    
