#!/usr/bin/env python3
import rospy
import math
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker
from scipy.ndimage import binary_dilation
from tf.transformations import euler_from_quaternion

class NodeAStar:
    def __init__(self, x, y, g=0, h=0, parent=None):
        self.x = x  # Linha no grid (Y)
        self.y = y  # Coluna no grid (X)
        self.g = g  # Custo acumulado
        self.h = h  # Heurística
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

class TurtlebotAStarStaticNav:
    def __init__(self):
        rospy.init_node('turtlebot3_astar_static', anonymous=True)

        # Metadados do Mapa
        self.grid = None
        self.resolution = None
        self.origin_x = None
        self.origin_y = None
        self.width = 0
        self.height = 0
        
        # Posição do Robô (Lida da Odometria perfeitamente alinhada via static TF)
        self.pose_x = None
        self.pose_y = None
        self.yaw = 0.0
        
        # Coordenadas do Alvo (Clicado no RViz)
        self.goal_real_x = None
        self.goal_real_y = None
        self.new_goal_received = False

        # Publishers e Subscribers
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.marker_publisher = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        
        # Inscrições nos tópicos (Voltando para o /odom)
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        
        self.rate = rospy.Rate(10) # 10Hz

        rospy.loginfo("Aguardando o mapa do map_server...")
        while self.grid is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
            
        rospy.loginfo("Aguardando inicialização da odometria...")
        while (self.pose_x is None or self.pose_y is None) and not rospy.is_shutdown():
            rospy.sleep(0.1)
            
        rospy.loginfo(">> TUDO PRONTO! Defina o destino clicando em '2D Nav Goal' no RViz.")

    def map_callback(self, msg):
        if self.grid is not None:
            return
        self.resolution = msg.info.resolution  
        self.origin_x = msg.info.origin.position.x 
        self.origin_y = msg.info.origin.position.y 
        self.width = msg.info.width
        self.height = msg.info.height
        
        raw_data = np.array(msg.data)
        grid_2d = raw_data.reshape((self.height, self.width))
        
        # Binarização inicial (Obstáculos e áreas desconhecidas viram 1)
        obstacle_mask = (grid_2d == 100) | (grid_2d == -1)
        
        # --- INFLAÇÃO ULTRA RÁPIDA COM SCIPY ---
        # Raio de 5 células * 5cm = 25cm de margem de segurança contra colisões
        radius = 5
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        circular_mask = x**2 + y**2 <= radius**2
        
        rospy.loginfo("Inflando obstáculos via SciPy binary_dilation (C-Space)...")
        inflated_mask = binary_dilation(obstacle_mask, structure=circular_mask)
        
        # Salva o grid final estruturado como lista Python
        self.grid = np.where(inflated_mask, 1, 0).tolist()
        rospy.loginfo(f"Grid de Navegação pronto: {self.width}x{self.height}")

    def odom_callback(self, msg):
        """ Captura a posição da odometria diretamente """
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, self.yaw) = euler_from_quaternion(orientation_list)

    def goal_callback(self, msg):
        """ Disparado ao clicar na ferramenta 2D Nav Goal do RViz """
        self.goal_real_x = msg.pose.position.x
        self.goal_real_y = msg.pose.position.y
        self.new_goal_received = True
        rospy.loginfo(f"Novo alvo recebido no frame MAP: X={self.goal_real_x:.2f}, Y={self.goal_real_y:.2f}")
        self.publish_goal_marker(self.goal_real_x, self.goal_real_y)

    def world_to_grid(self, real_x, real_y):
        """ Alinha as coordenadas em metros com os índices reais da matriz considerando a translação """
        col = int((real_x - self.origin_x) / self.resolution)
        row = int((real_y - self.origin_y) / self.resolution)
        
        # Garante confinamento dentro dos limites do array
        row = max(0, min(row, self.height - 1))
        col = max(0, min(col, self.width - 1))
        return row, col

    def grid_to_world(self, row, col):
        """ Converte índices do grid de volta para metros centralizando os pontos """
        real_x = (col * self.resolution) + self.origin_x + (self.resolution / 2.0)
        real_y = (row * self.resolution) + self.origin_y + (self.resolution / 2.0)
        return real_x, real_y

    def publish_goal_marker(self, x, y):
        """ Plota a esfera de debug vermelha no mapa do RViz """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.15
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.marker_publisher.publish(marker)

    def heuristic(self, r1, c1, r2, c2):
        return math.sqrt((r1 - r2)**2 + (c1 - c2)**2)

    def compute_astar(self, start_row, start_col, goal_row, goal_col):
        start_node = NodeAStar(start_row, start_col)
        goal_node = NodeAStar(goal_row, goal_col)
        
        # Validação se o clique caiu dentro da zona de inflação de segurança
        if self.grid[goal_row][goal_col] == 1:
            rospy.logwarn("Destino inválido: O ponto selecionado está dentro da margem de colisão de uma parede!")
            return None

        open_list = [start_node]
        closed_list = set()
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        while open_list:
            open_list.sort()
            current_node = open_list.pop(0)
            closed_list.add((current_node.x, current_node.y))
            
            if current_node.x == goal_node.x and current_node.y == goal_node.y:
                path = []
                while current_node:
                    path.append((current_node.x, current_node.y))
                    current_node = current_node.parent
                return path[::-1]
                
            for move in moves:
                nx, ny = current_node.x + move[0], current_node.y + move[1]
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    if self.grid[nx][ny] == 1 or (nx, ny) in closed_list:
                        continue
                        
                    move_cost = 1.414 if abs(move[0]) == 1 and abs(move[1]) == 1 else 1.0
                    g_cost = current_node.g + move_cost
                    h_cost = self.heuristic(nx, ny, goal_node.x, goal_node.y)
                    neighbor = NodeAStar(nx, ny, g_cost, h_cost, current_node)
                    
                    if any(op.x == nx and op.y == ny and op.g <= g_cost for op in open_list):
                        continue
                    open_list.append(neighbor)
        return None

    def navigate_to_waypoints(self, waypoints):
        idx = 0
        while not rospy.is_shutdown() and idx < len(waypoints):
            # Interrupção dinâmica se você clicar em outro ponto do RViz no meio do caminho
            if self.new_goal_received:
                return False

            target_x, target_y = waypoints[idx]
            distance = math.sqrt((target_x - self.pose_x)**2 + (target_y - self.pose_y)**2)
            angle_to_target = math.atan2(target_y - self.pose_y, target_x - self.pose_x)
            
            twist_msg = Twist()
            if distance < 0.22: # Raio de tolerância para aceitar o waypoint
                idx += 1
                continue
                
            angle_error = angle_to_target - self.yaw
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
            
            # Controle Proporcional Cinemático
            if abs(angle_error) > 0.4:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 1.3 * angle_error
            else:
                twist_msg.linear.x = min(0.15, 0.4 * distance)
                twist_msg.angular.z = 1.6 * angle_error
                
            self.velocity_publisher.publish(twist_msg)
            self.rate.sleep()
        
        self.velocity_publisher.publish(Twist())
        return True

    def run(self):
        while not rospy.is_shutdown():
            if self.new_goal_received:
                self.new_goal_received = False
                
                # Obtém posições convertendo os dados estáveis do /odom para o Grid
                start_row, start_col = self.world_to_grid(self.pose_x, self.pose_y)
                goal_row, goal_col = self.world_to_grid(self.goal_real_x, self.goal_real_y)
                
                rospy.loginfo(f"A* Planejando rota: Inicial [{start_row},{start_col}] -> Destino [{goal_row},{goal_col}]")
                grid_path = self.compute_astar(start_row, start_col, goal_row, goal_col)
                
                if grid_path:
                    # Converte o caminho de volta para metros e reduz densidade pegando de 5 em 5 células
                    waypoints = [self.grid_to_world(p[0], p[1]) for p in grid_path]
                    waypoints = waypoints[::5] + [waypoints[-1]]
                    
                    rospy.loginfo(f"Rota traçada com sucesso. Executando percurso...")
                    self.navigate_to_waypoints(waypoints)
                else:
                    rospy.logerr("Falha: O algoritmo A* não encontrou uma rota livre de colisões.")
            self.rate.sleep()

if __name__ == '__main__':
    try:
        navigator = TurtlebotAStarStaticNav()
        navigator.run()
    except rospy.ROSInterruptException:
        pass