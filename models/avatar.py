import os
import torch
from torch import nn
from torch.nn import functional as F
from pytorch3d.structures import Meshes
from skimage.io import imread
import numpy as np
import pickle as pkl

import src.networks as networks
from src.utils import harmonic_encoding
from src.utils.visuals import mask_errosion
from DECA.decalib.utils.renderer import SRenderY, set_rasterizer
from DECA.decalib.utils import util
from DECA.decalib.models import FLAME, encoders, decoders, decoders, lbs
from DECA.decalib.utils.config import cfg as deca_cfg


class Avatar(nn.Module):

    def __init__(self, cfg, device):
        super().__init__()
        
        self.device = device
        self.img_size = cfg.img_size
        model_cfg = deca_cfg.model

        self.n_param = model_cfg.n_shape+model_cfg.n_tex+model_cfg.n_exp+model_cfg.n_pose+model_cfg.n_cam+model_cfg.n_light
        self.n_detail = model_cfg.n_detail
        self.n_cond = model_cfg.n_exp + 3 # exp + jaw pose
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.n_exp, model_cfg.n_pose, model_cfg.n_cam, model_cfg.n_light]
        self.param_dict = {i:model_cfg.get('n_' + i) for i in model_cfg.param_list}        

        self.E_flame = encoders.ResnetEncoder(outsize=self.n_param)
        self.E_detail = encoders.ResnetEncoder(outsize=self.n_detail)
        # decoders
        self.flame = FLAME.FLAME(model_cfg)
        self.flametex = FLAME.FLAMETex(model_cfg)
        self.D_detail = decoders.Generator(
            latent_dim=self.n_detail+self.n_cond, 
            out_channels=1, 
            out_scale=model_cfg.max_z, 
            sample_mode = 'bilinear'
        )

        assert os.path.exists(cfg.paths.deca)
        print(f'Load DECA trained model weights: {cfg.paths.deca}')
        checkpoint = torch.load(cfg.paths.deca, map_location='cpu')
        util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
        util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
        util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])

        # Create the other modules
        deformer_input_ch = cfg.model.neural_texture_channels
        deformer_input_ch += 3
        deformer_input_ch += 3 * cfg.model.num_frequencies * 2
        output_channels = cfg.model.output_unet_deformer_feats
        input_mlp_feat = cfg.model.output_unet_deformer_feats + 2 * (1 + cfg.model.num_frequencies * 2)

        # Neural Texture Encoder
        self.autoencoder = networks.Autoencoder(
            cfg.autoencoder.autoenc_num_channels,
            cfg.autoencoder.autoenc_max_channels,
            cfg.autoencoder.autoenc_num_groups,
            cfg.autoencoder.autoenc_num_bottleneck_groups,
            cfg.autoencoder.autoenc_num_blocks,
            cfg.autoencoder.autoenc_num_layers,
            cfg.autoencoder.autoenc_block_type,
            input_channels=3 + 1,  # cat alphas
            input_size=cfg.model.model_image_size,
            output_channels=cfg.model.neural_texture_channels,
            norm_layer_type=cfg.model.norm_layer_type,
            activation_type=cfg.model.activation_type,
            conv_layer_type=cfg.model.conv_layer_type,
            use_psp=False,
        )

        # self.basis_deformer = None
        # self.vertex_deformer = None
        # self.mask_hard_threshold = 0.6

        # Unet deformer
        self.mesh_deformer = networks.UNet(
            cfg.unet.unet_num_channels,
            cfg.unet.unet_max_channels,
            cfg.unet.unet_num_groups,
            cfg.unet.unet_num_blocks,
            cfg.unet.unet_num_layers,
            cfg.unet.unet_block_type,
            input_channels=deformer_input_ch,
            output_channels=output_channels,
            skip_connection_type=cfg.unet.unet_skip_connection_type,
            norm_layer_type=cfg.model.norm_layer_type,
            activation_type=cfg.model.activation_type,
            conv_layer_type=cfg.model.conv_layer_type,
            downsampling_type='maxpool',
            upsampling_type='nearest',
        )

        # mlp_deformer
        self.mlp_deformer = networks.MLP(
            num_channels=256,
            num_layers=8,
            skip_layer=4,
            input_channels=input_mlp_feat,
            output_channels=3,
            activation_type=cfg.model.activation_type,
            last_bias=False,
        )

        # Neural renderer
        self.unet = networks.UNet(
            cfg.unet.unet_num_channels,
            cfg.unet.unet_max_channels,
            cfg.unet.unet_num_groups,
            cfg.unet.unet_num_blocks,
            cfg.unet.unet_num_layers,
            cfg.unet.unet_block_type,
            input_channels=cfg.model.neural_texture_channels + 3 * (
                    1 + cfg.unet.unet_use_vertex_cond) * (1 + 6 * 2),  # unet_use_normals_cond
            output_channels=3 + 1,
            skip_connection_type=cfg.unet.unet_skip_connection_type,
            norm_layer_type=cfg.model.norm_layer_type,
            activation_type=cfg.model.activation_type,
            conv_layer_type=cfg.model.conv_layer_type,
            downsampling_type='maxpool',
            upsampling_type='nearest',
        )

        rasterizer_type = 'pytorch3d'
        set_rasterizer(rasterizer_type)
        self.render = SRenderY(
            cfg.img_size, 
            obj_filename=cfg.paths.topology,
            files_dir=cfg.paths.addfiles_path,
            uv_size=cfg.uv_size, 
            rasterizer_type=rasterizer_type
        )

        grid_s = torch.linspace(0, 1, 224)
        v, u = torch.meshgrid(grid_s, grid_s)
        self.register_buffer('identity_grid_deca', torch.stack([u, v], dim=2)[None], persistent=False)

        grid_s = torch.linspace(-1, 1, cfg.img_size)
        v, u = torch.meshgrid(grid_s, grid_s)
        self.register_buffer('uv_grid', torch.stack([u, v], dim=2)[None])
        harmonized_uv_grid = harmonic_encoding.harmonic_encoding(self.uv_grid, num_encoding_functions=6)
        self.register_buffer('harmonized_uv_grid', harmonized_uv_grid[None])
        
        self.true_uvcoords = torch.load(cfg.paths.uvcoords).to(device)
        self.hair_list = pkl.load(open(cfg.paths.hair_list, 'rb'))
        self.neck_list = pkl.load(open(cfg.paths.neck_list, 'rb'))
        self.deforms_mask = torch.zeros(1, 5023, 1, device=device)
        self.mask_for_face = None

        if cfg.use_scalp_deforms:
            self.deforms_mask[:, self.hair_list] = 1.0
        if cfg.use_neck_deforms:
            self.deforms_mask[:, self.neck_list] = 1.0

        mask = imread(model_cfg.face_eye_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        mask = imread(model_cfg.face_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # displacement correction
        fixed_dis = np.load(model_cfg.fixed_displacement_path)
        self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # mean texture
        mean_texture = imread(model_cfg.mean_tex_path).astype(np.float32)/255.; mean_texture = torch.from_numpy(mean_texture.transpose(2,0,1))[None,:,:,:].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # dense mesh template, for save detail mesh
        self.dense_template = np.load(model_cfg.dense_template_path, allow_pickle=True, encoding='latin1').item()

        self.hair_mask = torch.zeros(1, 5023, 1)
        self.neck_mask = torch.zeros(1, 5023, 1)
        self.face_mask = torch.zeros(1, 5023, 1)

        self.hair_edge_list = pkl.load(open(cfg.paths.hair_edge_list, 'rb'))
        self.neck_edge_list = pkl.load(open(cfg.paths.neck_edge_list, 'rb'))

        def rm_from_list(a, b):
            return list(set(a) - set(b))

        # TODO save list to pickle
        hard_not_deform_list = [3587, 3594, 3595, 3598, 3600, 3630, 3634,
                                3635, 3636, 3637, 3643, 3644, 3646, 3649,
                                3650, 3652, 3673, 3676, 3677, 3678, 3679,
                                3680, 3681, 3685, 3691, 3693, 3695, 3697,
                                3698, 3701, 3703, 3707, 3709, 3713, 3371,
                                3372, 3373, 3374, 3375, 3376, 3377, 3378,
                                3379, 3382, 3383, 3385, 3387, 3389, 3392,
                                3393, 3395, 3397, 3399, 3413, 3414, 3415,
                                3416, 3417, 3418, 3419, 3420, 3421, 3422,
                                3423, 3424, 3441, 3442, 3443, 3444, 3445,
                                3446, 3447, 3448, 3449, 3450, 3451, 3452,
                                3453, 3454, 3455, 3456, 3457, 3458, 3459,
                                3460, 3461, 3462, 3463, 3494, 3496, 3510,
                                3544, 3562, 3578, 3579, 3581, 3583]
        exclude_list = [3382, 3377, 3378, 3379, 3375, 3374, 3544, 3494, 3496,
                        3462, 3463, 3713, 3510, 3562, 3372, 3373, 3376, 3371]

        hard_not_deform_list = list(rm_from_list(hard_not_deform_list, exclude_list))

        # if self.use_neck_deforms and self.external_params.get('updated_neck_mask', False):
        self.deforms_mask[:, hard_not_deform_list] = 0.0
        self.face_mask[:, self.hair_edge_list] = 0.0
        self.face_mask[:, self.neck_edge_list] = 0.0

        self.register_buffer('faces_hair_mask', util.face_vertices(self.hair_mask, self.render.faces))
        self.register_buffer('faces_neck_mask', util.face_vertices(self.neck_mask, self.render.faces))
        self.register_buffer('faces_face_mask', util.face_vertices(self.face_mask, self.render.faces))

        self.mask_hard_threshold = 0.6

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start+int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    def get_parametric_vertices(self, codedict, neutral_pose):
        cam_rot_mats = codedict['pose_rot_mats'][:, :1]
        batch_size = cam_rot_mats.shape[0]

        eye_rot_mats = neck_rot_mats = None

        if codedict['pose_rot_mats'].shape[1] >= 3:
            neck_rot_mats = codedict['pose_rot_mats'][:, 1:2]
            jaw_rot_mats = codedict['pose_rot_mats'][:, 2:3]
        else:
            jaw_rot_mats = codedict['pose_rot_mats'][:, 1:2]

        if codedict['pose_rot_mats'].shape[1] == 4:
            eye_rot_mats = codedict['pose_rot_mats'][:, 3:]

        # Use zero global camera pose inside FLAME fitting class
        cam_rot_mats_ = torch.eye(3).to(cam_rot_mats.device).expand(batch_size, 1, 3, 3)
        # Shaped vertices
        verts_neutral = self.flame.reconstruct_shape(codedict['shape'])

        # Visualize shape
        default_cam = torch.zeros_like(codedict['cam'])[:, :3]  # default cam has orthogonal projection
        default_cam[:, :1] = 5.0

        _, verts_neutral_frontal, _ = util.batch_orth_proj(verts_neutral, default_cam, flame=self.flame)
        shape_neutral_frontal = self.render.render_shape(verts_neutral, verts_neutral_frontal)

        # Apply expression and pose
        if neutral_pose:
            verts_parametric, rot_mats, root_joint, neck_joint = self.flame.reconstruct_exp_and_pose(
                verts_neutral, torch.zeros_like(codedict['exp']))
        else:
            verts_parametric, rot_mats, root_joint, neck_joint = self.flame.reconstruct_exp_and_pose(
                verts_neutral,
                codedict['exp'],
                cam_rot_mats_,
                neck_rot_mats,
                jaw_rot_mats,
                eye_rot_mats
            )

        # Add neck rotation
        if neck_rot_mats is not None:
            neck_rot_mats = neck_rot_mats.repeat_interleave(verts_parametric.shape[1], dim=1)
            verts_parametric = verts_parametric - neck_joint[:, None]
            verts_parametric = torch.matmul(neck_rot_mats.transpose(2, 3), verts_parametric[..., None])[..., 0]
            verts_parametric = verts_parametric + neck_joint[:, None]

        # Visualize exp verts
        _, verts_parametric_frontal, _ = util.batch_orth_proj(verts_parametric, default_cam, flame=self.flame)
        shape_parametric_frontal = self.render.render_shape(verts_parametric, verts_parametric_frontal)

        return cam_rot_mats, root_joint, verts_parametric, shape_neutral_frontal, shape_parametric_frontal
    
    def deform_source_mesh(self, verts_parametric, neural_texture):
        batch_size = verts_parametric.shape[0]

        verts_uvs = self.true_uvcoords[:, :, None, :2]  # 1 x V x 1 x 2

        verts_uvs = verts_uvs.repeat_interleave(batch_size, dim=0)

        # bs x 3 x H x W
        verts_texture = self.render.world2uv(verts_parametric) * 5

        enc_verts_texture = harmonic_encoding.harmonic_encoding(verts_texture.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        deform_unet_inputs = torch.cat([neural_texture.detach(), enc_verts_texture.detach()], dim=1)

        uv_deformations_codes = self.mesh_deformer(deform_unet_inputs)

        mlp_input_uv_z = F.grid_sample(uv_deformations_codes, verts_uvs, align_corners=False)[..., 0].permute(0, 2, 1)

        mlp_input_uv = F.grid_sample(self.uv_grid.repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2),
                                     verts_uvs, align_corners=False)[..., 0]
        mlp_input_uv = harmonic_encoding.harmonic_encoding(mlp_input_uv.permute(0, 2, 1), 6, )

        mlp_input_uv_deformations = torch.cat([mlp_input_uv_z, mlp_input_uv], dim=-1)

        if self.mask_for_face is None:
            self.mask_for_face = F.grid_sample((F.interpolate(self.uv_face_eye_mask.repeat(batch_size, 1, 1, 1)
                                                              , uv_deformations_codes.shape[-2:])),
                                               verts_uvs, align_corners=False)[..., 0].permute(0, 2, 1) > 0.5

        bs, v, ch = mlp_input_uv_deformations.shape
        deformation_project = self.mlp_deformer(mlp_input_uv_deformations.view(-1, ch))
        predefined_mask = None
        if predefined_mask is not None:
            deforms = torch.tanh(deformation_project.view(bs, -1, 3).contiguous())
            verts_empty_deforms = torch.zeros(batch_size, verts_uvs.shape[1], 3,
                                              dtype=verts_uvs.dtype, device=verts_uvs.device)
            verts_empty_deforms = verts_empty_deforms.scatter_(1, predefined_mask[None, :, None].expand(bs, -1, 3),
                                                               deforms)
            # self.deforms_mask.nonzero()[None].repeat(bs, 1, 1), deforms)
            verts_deforms = verts_empty_deforms
        else:
            verts_deforms = torch.tanh(deformation_project.view(bs, v, 3).contiguous())

        # if self.mask_for_face is not None and self.external_params.get('deform_face_tightness', 0.0) > 0.0:
            #      We slightly deform areas along the face
            # self.deforms_mask[self.mask_for_face[[0]]] = self.external_params.get('deform_face_tightness', 0.0)

        verts_deforms = verts_deforms * self.deforms_mask
        return verts_deforms

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
        ''' Convert displacement map into detail normal map
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()  # 1x3x224x224
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()  # 1x3x224x224
    
        uv_z = uv_z*self.uv_face_eye_mask
        uv_detail_vertices = uv_coarse_vertices + uv_z*uv_coarse_normals + self.fixed_uv_dis[None,None,:,:]*uv_coarse_normals.detach()
        dense_vertices = uv_detail_vertices.permute(0,2,3,1).reshape([batch_size, -1, 3])
        uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape([batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0,3,1,2)
        uv_detail_normals = uv_detail_normals*self.uv_face_eye_mask + uv_coarse_normals*(1-self.uv_face_eye_mask)
        return uv_detail_normals

    def forward(
        self, 
        img: torch.Tensor,
        mask: torch.Tensor,
        target_pose: torch.Tensor = False
    ):

        bs = img.shape[0]

        # Flame parameters
        img_flame = F.interpolate(img, (224, 224))
        flame_params = self.E_flame(img_flame)
        codes = self.decompose_code(flame_params, self.param_dict)
        codes['detail'] = self.E_detail(img_flame)

        pose = codes['pose'].view(bs, -1, 3)
        angle = torch.norm(pose + 1e-8, dim=2, keepdim=True)
        rot_dir = pose / angle
        codes['pose_rot_mats'] = lbs.batch_rodrigues(
            torch.cat([angle, rot_dir], dim=2).view(-1, 4)
        ).view(bs, pose.shape[1], 3, 3)  # cam & jaw | jaw | jaw & eyes

        # albedo
        albedo = self.flametex(codes['tex'])

        # Neural texture
        texture_inputs = torch.cat([img, mask], dim=1)
        neural_texture = self.autoencoder(texture_inputs)  # 1 x 8 x 224 x 224

        # Set output camera 
        default_cam = torch.zeros_like(codes['cam'])[:, :3]  # default cam has orthogonal projection
        default_cam[:, :1] = 5.0
 
        cam_rot_mats, root_joint, verts_template, \
        shape_neutral_frontal, shape_parametric_frontal = self.get_parametric_vertices(codes, False)

        verts_deforms = self.deform_source_mesh(verts_template, neural_texture) # B, 5023, 3

        # Obtain visualized frontal vertices
        faces = self.render.faces.expand(bs, -1, -1)

        vertex_normals = util.vertex_normals(verts_template, faces)  # B, 5023, 3
        verts_deforms = verts_deforms * vertex_normals

        original_mesh = Meshes(verts=verts_template.cpu(), faces=faces.long().cpu())

        verts_final = verts_template + verts_deforms

        _, verts_final_frontal, _ = util.batch_orth_proj(verts_final, default_cam, flame=self.flame)
        shape_final_frontal = self.render.render_shape(verts_final, verts_final_frontal)

        _, verts_target, landmarks_target = util.batch_orth_proj(
            verts_final.clone(), codes['cam'],
            root_joint, cam_rot_mats, self.flame)
        shape_target = self.render.render_shape(verts_final, verts_target)

        # _, verts_final_posed, _ = util.batch_orth_proj(verts_final.clone(), default_cam, flame=self.flame)
        # shape_final_posed = self.render.render_shape(verts_final, verts_final_posed)

        hair_neck_face_mesh_faces = torch.cat([self.faces_hair_mask, self.faces_neck_mask, self.faces_face_mask], dim=-1)
        render_output = self.render(
            verts_final, 
            verts_target, 
            albedo,
            codes['light'],
            face_masks=hair_neck_face_mesh_faces)

        rendered_texture = F.grid_sample(
            neural_texture, 
            render_output['uvcoords_images'].permute(0, 2, 3, 1)[..., :2], 
            mode='bilinear')
        rendered_texture = rendered_texture * render_output['alpha_images']
        flametex_texture = render_output['images']

        normals = render_output['normal_images'].permute(0, 2, 3, 1) # 1 x 3 x 224 x 224
        normal_inputs = harmonic_encoding.harmonic_encoding(normals, 6).permute(0, 3, 1, 2) # 1 x 39 x 224 x 224
        unet_inputs = torch.cat([rendered_texture, normal_inputs], dim=1) # 1 x 47 x 224 x 224
        unet_outputs = self.unet(unet_inputs)  # 1 x 4 x 224 x 224

        pred_img = torch.sigmoid(unet_outputs[:, :3])
        pred_unet_mask = torch.sigmoid(unet_outputs[:, 3:])

        mask_pred = pred_unet_mask[0].cpu() > self.mask_hard_threshold
        mask_pred = mask_errosion(mask_pred.float().numpy() * 255)
        pred_img_masked = pred_img[0].cpu() * (mask_pred) + (1 - mask_pred)

        # Render head model with details
        uv_z = self.D_detail(torch.cat([codes['pose'][:,3:], codes['exp'], codes['detail']], dim=1))  # 1 x 1 x 256 x 256
        uv_detail_normals = self.displacement2normal(uv_z, verts_final, render_output['normals'])
        uv_shading = self.render.add_SHlight(uv_detail_normals, codes['light'])
        uv_texture = albedo * uv_shading
        
        # shape_images, _, grid, alpha_images = self.render.render_shape(
        #     verts_final, 
        #     verts_target, 
        #     # h=self.image_size, 
        #     # w=self.image_size, 
        #     images=None, 
        #     return_grid=True)
        # detail_normal_images = F.grid_sample(uv_detail_normals, grid, align_corners=False)*alpha_images
        # shape_detail_images = self.render.render_shape(verts_final, verts_target, detail_normal_images=detail_normal_images, h=h, w=w, images=background)
        
        ## extract texture
        ## TODO: current resolution 256x256, support higher resolution, and add visibility
        uv_pverts = self.render.world2uv(verts_target)
        uv_gt = F.grid_sample(img, uv_pverts.permute(0,2,3,1)[:,:,:,:2], mode='bilinear')
        uv_texture_gt = uv_gt[:,:3,:,:]*self.uv_face_eye_mask + (uv_texture[:,:3,:,:]*(1-self.uv_face_eye_mask)) # 1x3x256x256
        # if self.cfg.model.use_tex:
            ## TODO: poisson blending should give better-looking results
        
        opdict = {
            'pred_img': pred_img,
            'pred_img_masked': pred_img_masked,
            'pred_unet_mask': pred_unet_mask,

            'verts': verts_final,
            'trans_verts': verts_target,
            'uv_texture': uv_texture,
            'normals': render_output['normals'],
            'normal_images': render_output['normal_images'],
            'uv_detail_normals': uv_detail_normals,
            'displacement_map': uv_z+self.fixed_uv_dis[None,None,:,:],
            'uv_texture_gt': uv_texture_gt
        }

        return opdict

    def save_obj(self, filename, opdict):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        i = 0
        vertices = opdict['verts'][i].detach().cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        texture = util.tensor2image(opdict['uv_texture_gt'][i])
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        normal_map = util.tensor2image(opdict['uv_detail_normals'][i]*0.5 + 0.5)
        util.write_obj(filename, vertices, faces, 
                        texture=texture, 
                        uvcoords=uvcoords, 
                        uvfaces=uvfaces, 
                        normal_map=normal_map)
        # upsample mesh, save detailed mesh
        texture = texture[:,:,[2,1,0]]
        normals = opdict['normals'][i].detach().cpu().numpy()
        displacement_map = opdict['displacement_map'][i].detach().cpu().numpy().squeeze()
        dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals, faces, displacement_map, texture, self.dense_template)
        util.write_obj(filename.replace('.obj', '_detail.obj'), 
                        dense_vertices, 
                        dense_faces,
                        colors = dense_colors,
                        inverse_face_order=True)