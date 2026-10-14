#version 450

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform ScenePush
{
   vec4 data;
} scene_push;

mat2 rotate2d(float angle)
{
   float c = cos(angle);
   float s = sin(angle);
   return mat2(c, -s, s, c);
}

float sphere_sdf(vec3 point, float radius)
{
   return length(point) - radius;
}

float rounded_box_sdf(vec3 point, vec3 bounds, float radius)
{
   vec3 distance_to_bounds = abs(point) - bounds;
   return length(max(distance_to_bounds, 0.0)) +
          min(max(distance_to_bounds.x,
                  max(distance_to_bounds.y, distance_to_bounds.z)),
              0.0) - radius;
}

float torus_sdf(vec3 point, vec2 radii)
{
   vec2 ring = vec2(length(point.xz) - radii.x, point.y);
   return length(ring) - radii.y;
}

float capsule_sdf(vec3 point, vec3 start, vec3 end, float radius)
{
   vec3 point_offset = point - start;
   vec3 axis = end - start;
   float along = clamp(dot(point_offset, axis) / dot(axis, axis), 0.0, 1.0);
   return length(point_offset - axis * along) - radius;
}

vec2 scene_sdf(vec3 point)
{
   float time = scene_push.data.x;
   vec2 nearest = vec2(point.y, 0.0);

   vec3 box_point = point -
      vec3(-1.05, 0.72 + 0.10 * sin(time * 0.83), 0.18);
   box_point.xz = rotate2d(time * 0.37) * box_point.xz;
   box_point.xy = rotate2d(0.28 + 0.15 * sin(time * 0.41)) * box_point.xy;
   float box_distance =
      rounded_box_sdf(box_point, vec3(0.42, 0.56, 0.34), 0.14);
   if (box_distance < nearest.x)
      nearest = vec2(box_distance, 1.0);

   vec3 torus_point = point -
      vec3(0.92, 0.78 + 0.08 * cos(time * 0.67), -0.28);
   torus_point.yz = rotate2d(0.75 + time * 0.31) * torus_point.yz;
   torus_point.xz = rotate2d(-0.24) * torus_point.xz;
   float torus_distance = torus_sdf(torus_point, vec2(0.48, 0.15));
   if (torus_distance < nearest.x)
      nearest = vec2(torus_distance, 2.0);

   vec3 capsule_point = point -
      vec3(0.05 + 0.18 * sin(time * 0.53), 1.18, -1.12);
   capsule_point.xz = rotate2d(-time * 0.26) * capsule_point.xz;
   float capsule_distance = capsule_sdf(
      capsule_point, vec3(-0.34, -0.42, 0.0), vec3(0.28, 0.50, 0.16), 0.20);
   float orbit_distance = sphere_sdf(
      capsule_point - vec3(0.52, 0.18 * sin(time), 0.12), 0.16);
   float sculpted_distance = min(capsule_distance, orbit_distance);
   if (sculpted_distance < nearest.x)
      nearest = vec2(sculpted_distance, 3.0);

   return nearest;
}

vec3 scene_normal(vec3 point)
{
   const float epsilon = 0.0025;
   const vec2 signs = vec2(1.0, -1.0) * 0.5773;
   return normalize(
      signs.xyy * scene_sdf(point + signs.xyy * epsilon).x +
      signs.yyx * scene_sdf(point + signs.yyx * epsilon).x +
      signs.yxy * scene_sdf(point + signs.yxy * epsilon).x +
      signs.xxx * scene_sdf(point + signs.xxx * epsilon).x);
}

float soft_shadow(vec3 origin, vec3 direction)
{
   float shadow = 1.0;
   float distance_along_ray = 0.035;
   for (int step = 0; step < 20; ++step) {
      float distance_to_scene =
         scene_sdf(origin + direction * distance_along_ray).x;
      shadow = min(shadow, 12.0 * distance_to_scene / distance_along_ray);
      distance_along_ray += clamp(distance_to_scene, 0.025, 0.34);
      if (distance_to_scene < 0.001 || distance_along_ray > 12.0)
         break;
   }
   return clamp(shadow, 0.12, 1.0);
}

vec3 material_color(float material, vec3 point)
{
   if (material < 0.5) {
      float checker = mod(floor(point.x * 1.35) + floor(point.z * 1.35), 2.0);
      return mix(vec3(0.055, 0.075, 0.10), vec3(0.10, 0.135, 0.17), checker);
   }
   if (material < 1.5)
      return vec3(0.92, 0.25, 0.10);
   if (material < 2.5)
      return vec3(0.08, 0.68, 0.82);
   return vec3(0.74, 0.24, 0.88);
}

void main()
{
   vec2 resolution = max(scene_push.data.yz, vec2(1.0));
   vec2 screen = gl_FragCoord.xy / resolution * 2.0 - 1.0;
   screen.x *= scene_push.data.w;
   screen.y = -screen.y;

   float time = scene_push.data.x;
   float camera_angle = 0.10 * sin(time * 0.19);
   vec3 ray_origin = vec3(4.25 * sin(camera_angle), 1.72,
                          4.25 * cos(camera_angle));
   vec3 target = vec3(0.0, 0.78, -0.20);
   vec3 forward = normalize(target - ray_origin);
   vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
   vec3 up = cross(forward, right);
   vec3 ray_direction =
      normalize(right * screen.x + up * screen.y + forward * 1.72);

   float distance_along_ray = 0.0;
   float material = -1.0;
   for (int step = 0; step < 56; ++step) {
      vec2 distance_and_material =
         scene_sdf(ray_origin + ray_direction * distance_along_ray);
      if (distance_and_material.x < 0.0015 * max(1.0, distance_along_ray)) {
         material = distance_and_material.y;
         break;
      }
      distance_along_ray += max(distance_and_material.x * 0.82, 0.008);
      if (distance_along_ray > 24.0)
         break;
   }

   vec3 sky_low = vec3(0.035, 0.07, 0.13);
   vec3 sky_high = vec3(0.24, 0.38, 0.52);
   vec3 color = mix(sky_low, sky_high, clamp(ray_direction.y * 0.5 + 0.5, 0.0, 1.0));
   vec3 light_direction = normalize(vec3(-0.55, 0.82, 0.38));
   color += vec3(1.0, 0.58, 0.28) *
            pow(max(dot(ray_direction, light_direction), 0.0), 96.0) * 0.9;

   if (material >= 0.0) {
      vec3 point = ray_origin + ray_direction * distance_along_ray;
      vec3 normal = scene_normal(point);
      float diffuse = max(dot(normal, light_direction), 0.0);
      float shadow = soft_shadow(point + normal * 0.012, light_direction);
      float hemisphere = 0.34 + 0.30 * max(normal.y, 0.0);
      float rim = pow(1.0 - max(dot(normal, -ray_direction), 0.0), 3.0);
      vec3 half_direction = normalize(light_direction - ray_direction);
      float specular = pow(max(dot(normal, half_direction), 0.0), 42.0);
      vec3 base = material_color(material, point);
      vec3 shaded = base * (hemisphere + 1.15 * diffuse * shadow);
      shaded += vec3(0.35, 0.56, 0.92) * rim * 0.24;
      shaded += vec3(1.0, 0.82, 0.62) * specular * shadow * 0.65;
      float fog = 1.0 - exp(-0.010 * distance_along_ray * distance_along_ray);
      color = mix(shaded, color, clamp(fog, 0.0, 0.88));
   }

   color = color / (color + vec3(0.72));
   color = pow(max(color, 0.0), vec3(0.4545));
   out_color = vec4(color, 1.0);
}
