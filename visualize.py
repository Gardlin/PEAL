import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torchvision.utils import make_grid, save_image
import cv2
import math
import copy
import  open3d as o3d
import time
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from plyfile import PlyData, PlyElement
def transform(pts, trans):
    """
    Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
    Input
        - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
    """
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0,2,1) + trans[:, :3, 3:4]
        return trans_pts.permute(0,2,1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T

def viz_supernode(pcd1,pcd2,supernode_clusters,name=None):
    "supernode_idx :[ n x m] m means culster_size"
    if isinstance(pcd1, torch.Tensor):
        pcd1 = pcd1.squeeze().detach().cpu().numpy()
        pcd2 = pcd2.squeeze().detach().cpu().numpy()
        supernode_clusters = supernode_clusters.detach().cpu().numpy()
    pcds=[]
    # pcd1=np.concatenate((pcd1,np.zeros((1,3))),axis=0)
    pcd2=np.concatenate((pcd2,np.zeros((1,3))),axis=0)
    pcd=np.concatenate((pcd1,pcd2),axis=0)
    cluster_pcd=pcd[supernode_clusters]
    cluster_pcd=make_open3d_point_cloud(cluster_pcd)
    cluster_pcd.paint_uniform_color([1,0,0])
    estimate_normal(cluster_pcd)
    pcds.append(cluster_pcd)
    frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
    cluster_idx=supernode_clusters[:]
    cluster_idx1=cluster_idx[cluster_idx<pcd1.shape[0]]
    cluster_idx2=cluster_idx[cluster_idx>pcd1.shape[0]-1]
    idx1=pcd1[cluster_idx1,:]
    idx2=pcd1[cluster_idx2-pcd1.shape[0],:]
    cluster_pcd1 = idx1#copy.deepcopy(idx1)
    cluster_pcd2 = idx2# copy.deepcopy(idx2)
    cluster_pcd1=make_open3d_point_cloud(idx1)
    cluster_pcd2=make_open3d_point_cloud(idx2)
    pcd1_o3d=make_open3d_point_cloud(pcd1)
    pcd2_o3d=make_open3d_point_cloud(pcd2)

    if not cluster_pcd1.has_normals():
        estimate_normal(cluster_pcd1)
        estimate_normal(cluster_pcd2)
        estimate_normal(pcd1_o3d)
        estimate_normal(pcd2_o3d)
    # color=[1, 0, 0]#np.random.random(3)
    cluster_pcd1.paint_uniform_color(color=[1,0,0])  # blue
    cluster_pcd2.paint_uniform_color(color=[1,0,0])  # blue
    pcd1_o3d.paint_uniform_color([1, 0.706, 0])
    pcd2_o3d.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([frame,cluster_pcd1,cluster_pcd2],window_name='cluster')
    # if idx2.shape[0]<1:
    #     o3d.visualization.draw_geometries([frame,cluster_pcd1,pcd1_o3d,pcd2_o3d],window_name='cluster')
    # elif idx1.shape[0]<1:
    #     o3d.visualization.draw_geometries([frame,cluster_pcd2,pcd1_o3d,pcd2_o3d],window_name='cluster')
    # else:
    #
    #     o3d.visualization.draw_geometries([frame,cluster_pcd1,cluster_pcd2,pcd1_o3d,pcd2_o3d],window_name='cluster')
    # pcds.append(cluster_pcd1)
    # pcds.append(cluster_pcd2)
    pcds.append(pcd1_o3d)
    pcds.append(pcd2_o3d)
    pcds.append(frame)
    o3d.visualization.draw_geometries([frame,cluster_pcd1,cluster_pcd2,pcd1_o3d,pcd2_o3d],window_name=name)


    # o3d.visualization.draw_geometries([pcds[i] for i in range(len(pcds))],window_name=name)

def integrate_trans(R, t):
    """
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4).to(R.device)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans

def reverse_normalize(coords,scale):
    coords=np.array(coords.detach().cpu())
    scale=np.array(scale.detach().cpu())
    # coords /=2.
    # coords +=0.5
    coords=coords+0.5
    coords=coords*scale
    return coords
def depth_img_show(img,gt_img,name):
    if isinstance(img, torch.Tensor):
        img=img.permute(1,2,0)
        gt_img=gt_img.permute(1,2,0)
        img = img.squeeze().detach().cpu().numpy()
        gt_img = gt_img.squeeze().detach().cpu().numpy()
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title(name)
    plt.subplot(1,2,2 )
    plt.imshow(gt_img)
    # plt.title(name)
    plt.show()
    time.sleep(3)
    plt.close()


def save_depth_img(img1,img2,name):
    if isinstance(img1, torch.Tensor):
        img1=img1.permute(1,2,0)
        img2=img2.permute(1,2,0)
        img1 = img1.squeeze().detach().cpu().numpy()
        img2 = img2.squeeze().detach().cpu().numpy()

    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.title(name[-20:])
    plt.subplot(1,2,2 )
    plt.imshow(img2)
    plt.savefig(f'{name}.png')
    # plt.show()



def estimate_normal(pcd, radius=0.06, max_nn=30):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
# def make_point_cloud(pts):
#     if isinstance(pts, torch.Tensor):
#         pts = pts.detach().cpu().numpy()
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pts)
#     return pcd

def make_open3d_point_cloud(xyz, color=None):
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def draw_ply(ply_path,Window_name=""):
    ply=o3d.io.read_point_cloud(ply_path)
    source_temp = copy.deepcopy(ply)
    if not source_temp.has_normals():
        estimate_normal(source_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])  # yellow
    o3d.visualization.draw_geometries([source_temp], window_name=Window_name)
def draw_pause(source,target,Window_name="",sec=3,Transformation=None,box1=None,box2=None):
    if Transformation==None:
        Transformation=np.identity(4)
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if not source_temp.has_normals():
        estimate_normal(source_temp)
        estimate_normal(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source.transform(Transformation)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=Window_name)
    frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
    if not box1==None:
        estimate_normal(box1)
        box1.paint_uniform_color([1, 0,0])
        vis.add_geometry(box1)
        if not box2==None:
            estimate_normal(box2)
            box2.paint_uniform_color([0, 0,0])
            vis.add_geometry(box2)

    vis.add_geometry(frame)
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    # vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    # vis.run()
    time.sleep(sec)
    vis.destroy_window()
def draw_pause_without_frame(source,target,Window_name="",sec=3,Transformation=None,box1=None,box2=None):
    if Transformation==None:
        Transformation=np.identity(4)
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if not source_temp.has_normals():
        estimate_normal(source_temp)
        estimate_normal(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source.transform(Transformation)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=Window_name)
    # frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
    if not box1==None:
        estimate_normal(box1)
        box1.paint_uniform_color([1, 0,0])
        vis.add_geometry(box1)
        if not box2==None:
            estimate_normal(box2)
            box2.paint_uniform_color([1, 0,0])
            vis.add_geometry(box2)

    # vis.add_geometry(frame)
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    # vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    # vis.run()
    time.sleep(sec)
    vis.destroy_window()
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):

    if intrinsic_image_dim == image_dim:
        return intrinsic

    intrinsic_return = np.copy(intrinsic)

    height_after = image_dim[1]
    height_before = intrinsic_image_dim[1]
    height_ratio = height_after / height_before

    width_after = image_dim[0]
    width_before = intrinsic_image_dim[0]
    width_ratio = width_after / width_before

    if width_ratio >= height_ratio:
        resize_height = height_after
        resize_width = height_ratio * width_before

    else:
        resize_width = width_after
        resize_height = width_ratio * height_before

    intrinsic_return[0,0] *= float(resize_width)/float(width_before)
    intrinsic_return[1,1] *= float(resize_height)/float(height_before)
    # account for cropping/padding here
    intrinsic_return[0,2] *= float(resize_width-1)/float(width_before-1)
    intrinsic_return[1,2] *= float(resize_height-1)/float(height_before-1)



    return intrinsic_return
# def draw_matplt_points(source, target, Transformation=None,Window_name="",box1=None,box2=None,font=None):
#     if type(source)==type(torch.tensor(0)):
#         source=source.detach().cpu().numpy()
#         target=target.detach().cpu().numpy()
#     import matplotlib.pyplot as plt
#     ax=plt.figure().add_subplot(projection='3d')
#     for
def return_o3d_normals(source, target, Transformation=None,Window_name="",box1=None,box2=None,font=None):
    if type(source)==type(torch.tensor(0)):
        source=source.detach().cpu().numpy()
        target=target.detach().cpu().numpy()
        source=make_open3d_point_cloud(source)
        target=make_open3d_point_cloud(target)
    if type(source)==type(np.array(0)):
        source=make_open3d_point_cloud(source)
        target=make_open3d_point_cloud(target)
    if not  type(Transformation)==type(np.array([0])):
        Transformation=np.identity(4)
    source_temp =source# copy.deepcopy(source)
    target_temp = target#copy.deepcopy(target)
    if not source_temp.has_normals():
        estimate_normal(source_temp)
        estimate_normal(target_temp)
    src_normal=np.array(source_temp.normals )# blue
    tgt_normal=np.array(target_temp.normals )# blue
    return src_normal,tgt_normal



def draw_registration_result(source, target, Transformation=None,Window_name="",box1=None,box2=None,font=None):
    if type(source)==type(torch.tensor(0)):
        source=source.detach().cpu().numpy()
        target=target.detach().cpu().numpy()
        source=make_open3d_point_cloud(source)
        target=make_open3d_point_cloud(target)
    if type(source)==type(np.array(0)):
        source=make_open3d_point_cloud(source)
        target=make_open3d_point_cloud(target)
    if not  type(Transformation)==type(np.array([0])):
        Transformation=np.identity(4)
    source_temp =source# copy.deepcopy(source)
    target_temp = target#copy.deepcopy(target)
    if not source_temp.has_normals():
        estimate_normal(source_temp)
        estimate_normal(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])  # blue
    target_temp.paint_uniform_color([0, 0.651, 0.929])  ##([0, 0.651, 0.929])
    source_temp.transform(Transformation)
    frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
    if box1==None:
      o3d.visualization.draw_geometries([frame,source_temp, target_temp],window_name=Window_name)
    # if not font==None:
    #     o3d.visualization.draw_geometries([frame,source_temp, target_temp],window_name=Window_name)
    else:
        box1.paint_uniform_color([1, 0,0])
        box2.paint_uniform_color([1, 0,0])
        o3d.visualization.draw_geometries([frame,source_temp, target_temp,box1,box2],window_name=Window_name)
def unproject(depth_img, depth_intrinsic, pose):
    " batchsize 240 320-->240 320"
    if isinstance(depth_img, torch.Tensor):
        depth_img = depth_img.squeeze(dim=0).numpy()
    depth_shift = 1000.0
    x, y = np.meshgrid(np.linspace(0, depth_img.shape[1] - 1, depth_img.shape[1]),
                       np.linspace(0, depth_img.shape[0] - 1, depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:, :, 0] = x
    uv_depth[:, :, 1] = y
    uv_depth[:, :, 2] = depth_img / depth_shift
    uv_depth = np.reshape(uv_depth, [-1, 3])
    uv_depth = uv_depth[np.where(uv_depth[:, 2] != 0), :].squeeze()
    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    bx = depth_intrinsic[0, 3]
    by = depth_intrinsic[1, 3]
    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    bx = depth_intrinsic[0, 3]
    by = depth_intrinsic[1, 3]
    point_list = []
    n = uv_depth.shape[0]
    points = np.ones((n, 4))
    X = (uv_depth[:, 0] - cx) * uv_depth[:, 2] / fx + bx
    Y = (uv_depth[:, 1] - cy) * uv_depth[:, 2] / fy + by
    points[:, 0] = X
    points[:, 1] = Y
    points[:, 2] = uv_depth[:, 2]
    inds = points[:, 2] > 0
    points = points[inds, :]
    points_world = np.dot(points, np.transpose(pose))
    return points_world[:, :3]
def save_ply(pcd_path,color=None, save_name='pcd_read.ply'):
    if (type(pcd_path)) == str:
        data = np.load(pcd_path)['pcd']
    else:
        data = pcd_path
    if not color==None:
        color=np.array(color)
        colors=[(color[i, 0], color[i, 1], color[i, 2]) for i in range(color.shape[0])]
        vertices = np.empty(color.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        # vertex_color=np.array(colors,dtype=[('red','u1'),('green','u1'),('blue','u1')])
        # colors_el=PlyElement.describe(vertex_color,'color',comments=['colors'])
        # data_point = [(data[i, 0], data[i, 1], data[i, 2]) for i in range(data.shape[0])]
        x,y,z=data[:,0],data[:,1],data[:,2]
        red,green,blue=color[:,0],color[:,1],color[:,2]
        vertices['x'] = x.astype('f4')
        vertices['y'] = y.astype('f4')
        vertices['z'] = z.astype('f4')
        vertices['red'] = red.astype('u1')
        vertices['green'] = green.astype('u1')
        vertices['blue'] = blue.astype('u1')
        # vertex = np.array(data_point, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        # # data_color=[(data[i,3],data[i,4],data[i,5]) for i in range(data.shape[0])]
        # # data_color=np.array(data_color,dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        # # color=PlyElement.describe(data_color,'color', comments=['colors'])
        # el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
        ply.write(f"{save_name}.ply")
        # PlyData([el,colors_el], text=True).write(f"{save_name}.ply")
    else:
        data_point = [(data[i, 0], data[i, 1], data[i, 2]) for i in range(data.shape[0])]
        vertex = np.array(data_point, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        # data_color=[(data[i,3],data[i,4],data[i,5]) for i in range(data.shape[0])]
        # data_color=np.array(data_color,dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        # color=PlyElement.describe(data_color,'color', comments=['colors'])

        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el], text=True).write(f"{save_name}.ply")

    print(save_name)

def normalize_pcd(pcd, flag=True):
    "defaul pcd: numpy ndarray"
    # print("Loading point cloud")
    point_cloud = np.array(pcd).astype(np.float32)
    # print("Finished loading point cloud")

    coords = point_cloud[:, :3]

    # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
    # sample efficiency)
    coords_center = np.mean(coords, axis=0, keepdims=True)
    coords -= coords_center
    if flag:
        coord_max = np.amax(coords)
        coord_min = np.amin(coords)
    else:
        coord_max = np.amax(coords, axis=0, keepdims=True)
        coord_min = np.amin(coords, axis=0, keepdims=True)
    scale = (coord_max - coord_min)
    coords = (coords - coord_min)
    coords = coords / scale
    coords -= 0.5
    # coords *= 2.
    return coords, scale, coords_center, coord_min


def occluded_normalize(pcd, gt_center, gt_scale, gt_min):
    point_cloud = np.array(pcd).astype(np.float32)
    coords = point_cloud[:, :3]
    coords -= gt_center
    coords -= gt_min
    coords = coords / gt_scale
    coords -= 0.5
    return coords, gt_scale, gt_center, 0.0


def down_sample(pcd, max_points):
    if pcd.shape[0] < max_points:
        return pcd, 0
    idx = np.random.permutation(pcd.shape[0])[:max_points]
    # pcd = pcd[idx]
    return idx


def get_occluded_pcd(path):
    data=torch.load(path)
    occluded_pcd=data['pcd1']
    gt=data['no_occlu_pcd']
    coords, gt_center, gt_scale, gt_min=normalize_pcd(gt.detach().cpu().numpy())
    pcd_coords, scale, coords_center, coord_min=occluded_normalize(occluded_pcd, gt_center, gt_scale, gt_min)
    save_ply(pcd_coords,'no_downsample.ply')

def viz_result():
    viz_list=['/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/325.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/350.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/375.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/400.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/425.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/450.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/475.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/50.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/500.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/525.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/550.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/575.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/600.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/625.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/650.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/675.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/700.npz',
              '/home/lab507/PycharmProjects/pythonProject/result/target//scene0010_01/pcd/725.npz']
    for fname in viz_list:
        pcd = get_occluded_pcd(fname)
        save_ply('pcd',fname)

# viz_result()