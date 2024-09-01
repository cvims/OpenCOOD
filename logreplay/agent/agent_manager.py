import carla
import random
import logging
import os
from opencood.hypes_yaml.yaml_utils import save_yaml_wo_overwriting


class AgentManager(object):
    def __init__(self, client, world, config, output_root, scene_name) -> None:
        self.client = client
        self.world = world
        self.carla_map = world.get_map()
        self.center = None
        self.out_root = output_root
        self.scene_name = scene_name

        # whether to activate this module
        self.activate = config['activate']
        if not self.activate:
            return
        
        # save flags
        self.save_yml = config['save_yml']
        self.save_all_agent_positions = config['save_all_agent_positions']

        self.agent_list = []

        for agent_content in config['agent_list']:
            agent_name = agent_content['name']
            agent_args = agent_content['args']

            if agent_name == 'vehicle':
                self.spawn_vehicles(agent_args)
            elif agent_name == 'walker':
                self.spawn_walkers(agent_args)
            else:
                raise ValueError(f'Agent {agent_name} not recognized')


    def set_random_vehicle_blueprint_attributes(self, blueprint):
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)

        blueprint.set_attribute('role_name', 'autopilot')

        return blueprint


    def spawn_vehicles(self, agent_args):
        spawn_count = agent_args['spawn_count']

        vehicle_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()

        # check if number of spawn points is less than number of vehicles
        if len(spawn_points) < spawn_count:
            print('Number of spawn points is less than number of vehicles')

        spawn_count = min(len(spawn_points), spawn_count)

        # random permutation of spawn points
        random.shuffle(spawn_points)
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        batch = []
        for _, spawn_point in zip(range(spawn_count), spawn_points):
            blueprint = random.choice(vehicle_blueprints)
            blueprint = self.set_random_vehicle_blueprint_attributes(blueprint)
            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, spawn_point)
                .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port())))
        
        spawned_actor_ids = []
        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                logging.info(response.error)
            else:
                spawned_actor_ids.append(response.actor_id)

        print(f'Spawned {len(spawned_actor_ids)} vehicles')
        
        # add vehicles to agents list
        self.agent_list.extend([self.world.get_actor(actor_id) for actor_id in spawned_actor_ids])


    def spawn_walkers(self, agent_args):
        spawn_count = agent_args['spawn_count']

        walkers = []

        blueprint_library = self.world.get_blueprint_library()
        walker_blueprints = blueprint_library.filter('walker.*')

        spawn_points = []
        for _ in range(spawn_count):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc != None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        batch = []
        walker_speed = []

        SpawnActor = carla.command.SpawnActor

        for spawn_point in spawn_points:
            walker_bp = random.choice(walker_blueprints)
            # set as not invincible
            probability = random.randint(0,100 + 1)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            if walker_bp.has_attribute('can_use_wheelchair') and probability < 11:
                walker_bp.set_attribute('use_wheelchair', 'true')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > 0.05):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)

            batch.append(SpawnActor(walker_bp, spawn_point))

        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        all_id = []
        for i in range(len(results)):
            if results[i].error:
                logging.info(results[i].error)
            else:
                walkers.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])

        walker_speed = walker_speed2

        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers[i]["id"]))

        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.info(results[i].error)
            else:
                walkers[i]["con"] = results[i].actor_id

        for i in range(len(walkers)):
            all_id.append(walkers[i]["con"])
            all_id.append(walkers[i]["id"])

        all_actors = self.world.get_actors(all_id)

        self.world.tick()

        self.world.set_pedestrians_cross_factor(0.05)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        self.world.tick()
        
        print(f'Spawned {len(all_id) / 2} walkers')

        # add walkers to agents list
        spawned_walkers = [self.world.get_actor(walker['id']) for walker in walkers]
        self.agent_list.extend(spawned_walkers)


    def run_step(self, timestamp, vehicle_dict):
        self.current_timstamp = timestamp
        self.data_dump(vehicle_dict)

    
    def format_agent_attributes(self, agent):
        _transform = agent.get_transform()
        _bbx = agent.bounding_box
        _loc = agent.get_location()
        angle = [_transform.rotation.roll, _transform.rotation.yaw, _transform.rotation.pitch]
        center =  [_bbx.location.x, _bbx.location.y, _bbx.location.z]
        extent = [_bbx.extent.x, _bbx.extent.y, _bbx.extent.z]
        location = [_loc.x, _loc.y, _loc.z]

        return dict(
            angle=angle,
            center=center,
            extent=extent,
            location=location
        )


    def data_dump(self, vehicle_dict):
        if not self.activate:
            return

        if self.save_all_agent_positions:
            # use also preinitialized agents (replay)
            # get all agents from carla world
            all_agents = self.world.get_actors()
        else:
            all_agents = self.agent_list

        vehicle_agents = {}
        walker_agents = {}

        for agent in all_agents:
            # get correct initial actor id through the vehicle dict
            agent_id = agent.id
            for veh_id, value in vehicle_dict.items():
                if value['actor_id'] == agent_id:
                    agent_id = veh_id
                    break

            agent_id = int(agent_id)
            if 'vehicle' in agent.type_id:
                vehicle_agents[agent_id] = self.format_agent_attributes(agent)
            elif 'walker' in agent.type_id:
                walker_agents[agent_id] = self.format_agent_attributes(agent)
        
        all_agents = dict(
            vehicles=vehicle_agents,
            walkers=walker_agents
        )
    
        if self.save_yml:
            # save metadata
            save_yaml_name = os.path.join(
                self.out_root,
                self.current_timstamp + '_all_agents.yaml')
            
            save_yaml_wo_overwriting(
                all_agents, save_yaml_name
            )