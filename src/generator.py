"""
Knowledge Causality Generator — Task 1 of VBVR-3D.

Task: Given a ball at the top of a slope about to roll into a tower,
predict the collision outcome based on the ball's material (heavy iron vs
lightweight plastic).

Modify this file to implement a different 3D task.
"""

import os
import math
import random
import mathutils

from core.base_blender_generator import BaseBlenderGenerator
from core.schemas import TaskPair
from src.prompts import get_prompt


class CausalityGenerator(BaseBlenderGenerator):
    """
    Generates the 'Material-Momentum Causality' task.

    Each sample randomises:
      - Ball type  : heavy iron (50 kg) OR lightweight plastic (0.4 kg)
      - Slope angle: 14° – 31°
      - Tower floors: 3 – 8 blocks
      - Ball radius : 0.35 – 0.55 m
      - Camera orbit: slight side-to-side shift for visual variety

    Output per sample:
      first_frame.png   — pre-collision scene (ball at top of slope)
      ground_truth.mp4  — physics simulation: ball rolls down and hits tower
      prompt.txt        — natural-language task question
      metadata.json     — all randomised parameters
    """

    # ── Material helpers ──────────────────────────────────────────────────────

    def _sky_world(self):
        """Physical sky for realistic ambient lighting."""
        world = self.bpy.data.worlds.new("Sky")
        self.bpy.context.scene.world = world
        world.use_nodes = True
        wnt = world.node_tree; wnt.nodes.clear()
        bg  = wnt.nodes.new('ShaderNodeBackground')
        sky = wnt.nodes.new('ShaderNodeTexSky')
        out = wnt.nodes.new('ShaderNodeOutputWorld')
        try: sky.sky_type = 'HOSEK_WILKIE'
        except Exception: pass
        sky.sun_elevation = math.radians(35)
        sky.turbidity     = 3.0
        bg.inputs['Strength'].default_value = 1.2
        wnt.links.new(sky.outputs['Color'],      bg.inputs['Color'])
        wnt.links.new(bg.outputs['Background'], out.inputs['Surface'])

    def _mat_concrete(self, name):
        m = self.bpy.data.materials.new(name); m.use_nodes = True
        nt = m.node_tree; nt.nodes.clear()
        b = nt.nodes.new('ShaderNodeBsdfPrincipled')
        o = nt.nodes.new('ShaderNodeOutputMaterial')
        n = nt.nodes.new('ShaderNodeTexNoise')
        n.inputs['Scale'].default_value = 8.0; n.inputs['Detail'].default_value = 10.0
        r = nt.nodes.new('ShaderNodeValToRGB')
        r.color_ramp.elements[0].color = (0.55, 0.53, 0.50, 1)
        r.color_ramp.elements[1].color = (0.70, 0.68, 0.65, 1)
        nt.links.new(n.outputs['Fac'], r.inputs['Fac'])
        nt.links.new(r.outputs['Color'], b.inputs['Base Color'])
        b.inputs['Roughness'].default_value = 0.85
        nt.links.new(b.outputs['BSDF'], o.inputs['Surface']); return m

    def _mat_worn_metal(self, name):
        m = self.bpy.data.materials.new(name); m.use_nodes = True
        nt = m.node_tree; nt.nodes.clear()
        b = nt.nodes.new('ShaderNodeBsdfPrincipled')
        o = nt.nodes.new('ShaderNodeOutputMaterial')
        b.inputs['Base Color'].default_value = (0.35, 0.30, 0.28, 1)
        b.inputs['Metallic'].default_value = 0.9; b.inputs['Roughness'].default_value = 0.4
        nt.links.new(b.outputs['BSDF'], o.inputs['Surface']); return m

    def _mat_rusted_iron(self, name, hue=0.0):
        m = self.bpy.data.materials.new(name); m.use_nodes = True
        nt = m.node_tree; nt.nodes.clear()
        b = nt.nodes.new('ShaderNodeBsdfPrincipled')
        o = nt.nodes.new('ShaderNodeOutputMaterial')
        n = nt.nodes.new('ShaderNodeTexNoise')
        n.inputs['Scale'].default_value = 6.0; n.inputs['Detail'].default_value = 8.0
        rc = nt.nodes.new('ShaderNodeValToRGB')
        rc.color_ramp.elements[0].color = (max(0, .20+hue*.3), max(0, .08+hue*.1), .03, 1)
        rc.color_ramp.elements[1].color = (max(0, .35+hue*.3), max(0, .18+hue*.1), .08, 1)
        rr = nt.nodes.new('ShaderNodeValToRGB')
        rr.color_ramp.elements[0].color = (.45,.45,.45,1)
        rr.color_ramp.elements[1].color = (.75,.75,.75,1)
        nt.links.new(n.outputs['Fac'], rc.inputs['Fac'])
        nt.links.new(n.outputs['Fac'], rr.inputs['Fac'])
        nt.links.new(rc.outputs['Color'], b.inputs['Base Color'])
        nt.links.new(rr.outputs['Color'], b.inputs['Roughness'])
        b.inputs['Metallic'].default_value = 0.85
        nt.links.new(b.outputs['BSDF'], o.inputs['Surface']); return m

    def _mat_plastic(self, name):
        m = self.bpy.data.materials.new(name); m.use_nodes = True
        nt = m.node_tree; nt.nodes.clear()
        b = nt.nodes.new('ShaderNodeBsdfPrincipled')
        o = nt.nodes.new('ShaderNodeOutputMaterial')
        b.inputs['Base Color'].default_value          = (0.8, 0.9, 1.0, 1)
        b.inputs['Roughness'].default_value           = 0.08
        b.inputs['Transmission Weight'].default_value = 0.95
        b.inputs['IOR'].default_value                 = 1.45
        nt.links.new(b.outputs['BSDF'], o.inputs['Surface']); return m

    def _mat_wood(self, name):
        m = self.bpy.data.materials.new(name); m.use_nodes = True
        nt = m.node_tree; nt.nodes.clear()
        b = nt.nodes.new('ShaderNodeBsdfPrincipled')
        o = nt.nodes.new('ShaderNodeOutputMaterial')
        w = nt.nodes.new('ShaderNodeTexWave')
        w.wave_type = 'BANDS'; w.inputs['Scale'].default_value = 5.0
        w.inputs['Distortion'].default_value = 3.5; w.inputs['Detail'].default_value = 8.0
        r = nt.nodes.new('ShaderNodeValToRGB')
        r.color_ramp.elements[0].color = (0.45,0.25,0.08,1)
        r.color_ramp.elements[1].color = (0.72,0.45,0.18,1)
        nt.links.new(w.outputs['Fac'], r.inputs['Fac'])
        nt.links.new(r.outputs['Color'], b.inputs['Base Color'])
        b.inputs['Roughness'].default_value = 0.6
        nt.links.new(b.outputs['BSDF'], o.inputs['Surface']); return m

    # ── Rigid-body helper ──────────────────────────────────────────────────────

    def _add_rb(self, obj, body_type='ACTIVE', mass=1.0, friction=0.5, restitution=0.3):
        with self.bpy.context.temp_override(active_object=obj, selected_objects=[obj]):
            self.bpy.ops.rigidbody.object_add()
        obj.rigid_body.type        = body_type
        obj.rigid_body.mass        = mass
        obj.rigid_body.friction    = friction
        obj.rigid_body.restitution = restitution

    # ── Main task generator ────────────────────────────────────────────────────

    def generate_task_pair(self, task_id: str) -> TaskPair:
        self.clear_scene()

        # ── Randomised parameters ──────────────────────────────────────────
        is_heavy   = random.choice([True, False])
        slope_ang  = random.uniform(0.25, 0.55)
        n_floors   = random.randint(3, 8)
        ball_r     = random.uniform(0.35, 0.55)
        hue_shift  = random.uniform(0.0, 0.5)
        cam_side   = random.uniform(-0.5, 0.5)
        # ──────────────────────────────────────────────────────────────────

        self._sky_world()
        self.bpy.ops.rigidbody.world_add()

        # Ground
        self.bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
        ground = self.bpy.context.object
        ground.data.materials.append(self._mat_concrete(f"Concrete_{task_id}"))
        self._add_rb(ground, 'PASSIVE', friction=0.5, restitution=0.2)

        # Slope
        self.bpy.ops.mesh.primitive_cube_add(size=2, location=(-5, 0, 1))
        slope = self.bpy.context.object
        slope.scale = (2.5, 1.0, 0.08)
        slope.rotation_euler = (0, -slope_ang, 0)
        slope.data.materials.append(self._mat_worn_metal(f"Metal_{task_id}"))
        self._add_rb(slope, 'PASSIVE', friction=0.08, restitution=0.1)

        # Ball
        start_z = 1.0 + 2.5 * math.sin(slope_ang) + ball_r + 0.3
        self.bpy.ops.mesh.primitive_uv_sphere_add(
            radius=ball_r, location=(-7.5, 0, start_z), segments=48, ring_count=24)
        ball = self.bpy.context.object
        self.bpy.ops.object.shade_smooth()
        mass_val = 50.0 if is_heavy else 0.4
        rest_val = 0.05 if is_heavy else 0.85
        ball.data.materials.append(
            self._mat_rusted_iron(f"Ball_{task_id}", hue_shift) if is_heavy
            else self._mat_plastic(f"Ball_{task_id}")
        )
        self._add_rb(ball, 'ACTIVE', mass=mass_val, friction=0.2, restitution=rest_val)

        # Tower
        mat_wood = self._mat_wood(f"Wood_{task_id}")
        bs, bg = 0.38, 0.02
        for i in range(n_floors):
            z = (bs + bg) * i + bs / 2
            self.bpy.ops.mesh.primitive_cube_add(size=bs, location=(0, 0, z))
            blk = self.bpy.context.object
            blk.data.materials.append(mat_wood)
            self._add_rb(blk, 'ACTIVE', mass=1.0, friction=0.8)

        # Fill light
        self.bpy.ops.object.light_add(type='AREA', location=(-3, -4, 10))
        fl = self.bpy.context.object
        fl.data.energy = 800; fl.data.size = 6.0
        fl.rotation_euler = (math.radians(20), 0, math.radians(-20))

        # Camera (auto-aim)
        cam_pos    = mathutils.Vector((-2.0 + cam_side, -13.0, 6.0))
        cam_target = mathutils.Vector((-3.0, 0.0, 1.5))
        self.bpy.ops.object.camera_add(location=cam_pos)
        cam = self.bpy.context.object
        self.bpy.context.scene.camera = cam
        cam.data.lens = 35
        cam.rotation_euler = (cam_target - cam_pos).to_track_quat('-Z', 'Y').to_euler()

        # ── Output paths ───────────────────────────────────────────────────
        output_dir = os.path.join(
            str(self.config.output_dir),
            f"{self.config.domain}_task", task_id)
        os.makedirs(output_dir, exist_ok=True)

        first_frame_path = os.path.join(output_dir, "first_frame.png")
        video_path       = os.path.join(output_dir, "ground_truth.mp4")

        # ── Render ─────────────────────────────────────────────────────────
        self.render_first_frame(first_frame_path)
        self.render_video(video_path, bake_physics=True)

        # ── Prompt ─────────────────────────────────────────────────────────
        ball_type = "iron" if is_heavy else "plastic"
        prompt = get_prompt(
            ball_type=ball_type,
            tower_floors=n_floors,
        )

        # ── Metadata ──────────────────────────────────────────────────────
        task_data = {
            "is_heavy_ball":   is_heavy,
            "ball_type":       ball_type,
            "slope_angle_deg": round(math.degrees(slope_ang), 1),
            "tower_floors":    n_floors,
            "ball_radius":     round(ball_r, 3),
            "ball_mass_kg":    mass_val,
        }

        return TaskPair(
            task_id=task_id,
            domain=self.config.domain,
            prompt=prompt,
            first_image=first_frame_path,
            ground_truth_video=video_path if os.path.exists(video_path) else None,
            metadata=self._build_metadata(task_id, task_data),
        )
