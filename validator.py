
# validator.py
import math

class SimpleValidator:
    """
    Validador simples para estudo.

    Funções:
    - Matching básico por distância (associa detecções entre frames).
    - Classificação por área: "car" ou "truck".
    - Contagem de veículos que ENTRAM pela borda superior da ROI:
        * Se o centróide cruza de cima (prev_y < 5) para dentro (cy > 10), conta.
        * Para novos objetos já "dentro" (cy > 10 no primeiro frame), também conta.

    Observações:
    - Não altera sua pipeline (ROI + contornos + MOG2).
    - Retorna (tipo, counted, id) no register() para permitir salvar a imagem.
    """

    def __init__(self, min_area, truck_area_threshold=5000):
        """
        Parâmetros:
            min_area: área mínima para considerar um contorno como veículo (evita ruído).
            truck_area_threshold: área a partir da qual é considerado "truck".
                                  (ajuste conforme seu vídeo/ROI; 5000 é um bom ponto de partida)
        """
        self.min_area = min_area
        self.truck_area_threshold = truck_area_threshold

        # Armazena a última posição (cx, cy) dos objetos rastreados de forma simples
        # Mapeamento: object_id -> (cx, cy)
        self.objects = {}

        # ID incremental para novos objetos
        self.next_id = 1

        # Contadores totais
        self.cars = 0
        self.trucks = 0

    # -----------------------------
    # Matching simples por distância
    # -----------------------------
    def _match(self, cx, cy):
        """
        Tenta associar (cx, cy) ao objeto existente mais próximo,
        desde que esteja a menos de 50 px de distância.
        Retorna: object_id correspondente, ou None se não houver match.
        """
        best_id = None
        best_dist = 9999.0

        for oid, (px, py) in self.objects.items():
            dist = math.hypot(cx - px, cy - py)
            if dist < best_dist and dist < 50:  # tolerância
                best_dist = dist
                best_id = oid

        return best_id

    # -----------------------------
    # Classificação por área
    # -----------------------------
    def _type_by_area(self, area):
        """
        Define o tipo do veículo com base na área do contorno.
        """
        return "truck" if area >= self.truck_area_threshold else "car"

    # -----------------------------
    # Registro principal por frame
    # -----------------------------
    def register(self, cx, cy, area):
        """
        Registra uma detecção em um frame.

        Entradas:
            cx, cy: centróide da detecção (coordenadas relativas à ROI)
            area  : área do contorno

        Retorna:
            (tipo, counted, object_id)
              - tipo: "car", "truck" ou "ignore"
              - counted: True se gerou contagem nesse frame
              - object_id: ID numérico do objeto no matching simples
        """
        # Ignorar ruídos
        if area < self.min_area:
            return "ignore", False, None

        # Tenta associar com algum objeto existente
        oid = self._match(cx, cy)

        # Caso seja um novo objeto
        if oid is None:
            oid = self.next_id
            self.objects[oid] = (cx, cy)
            self.next_id += 1

            # Checagem de ENTRADA na criação do objeto:
            # centróide já "dentro" da ROI?
            if cy > 10:  # tolerância baixa para borda superior
                vtype = self._type_by_area(area)
                if vtype == "truck":
                    self.trucks += 1
                else:
                    self.cars += 1
                return vtype, True, oid

            # Ainda não entrou (está acima da borda)
            return self._type_by_area(area), False, oid

        # Atualiza objeto existente
        prev_y = self.objects[oid][1]
        self.objects[oid] = (cx, cy)

        # Cruzou a borda superior agora?
        if prev_y < 5 and cy > 10:
            vtype = self._type_by_area(area)
            if vtype == "truck":
                self.trucks += 1
            else:
                self.cars += 1
            return vtype, True, oid

        # Sem contagem neste frame
        return self._type_by_area(area), False, oid

    # -----------------------------
    # Totais
    # -----------------------------
    def get_counts(self):
        """
        Retorna a quantidade total de carros e caminhões que ENTRARAM.
        """
        return self.cars, self.trucks
