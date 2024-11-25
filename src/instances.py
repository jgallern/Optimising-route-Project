import pandas as pd
import matplotlib.pyplot as plt
import os


class instances:
    def __init__(self, file):
        self.file = file
        self.name = os.path.basename(file).strip(".txt")
        # opening the salomon instances txt file
        with open(self.file, "r") as file:
            lines = file.readlines()

        # Séparer les sections
        vehicle_section = []
        customer_section = []
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("VEHICLE"):
                current_section = vehicle_section
            elif line.startswith("CUSTOMER"):
                current_section = customer_section
            elif current_section is not None and line:
                current_section.append(line)

        # Parse vehicle data
        vehicle_headers = vehicle_section[0].split()  # Extraction des colonnes
        vehicle_data = [list(map(int, line.split())) for line in vehicle_section[1:]]
        self.vehicle_df = pd.DataFrame(vehicle_data, columns=vehicle_headers)

        ## Parse customer data
        customer_headers = ['CUST NO.', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY TIME', 'DUE DATE', 'SERVICE TIME']
        Deposit = customer_section[1]
        customer_data = [list(map(float, line.split())) for line in customer_section[1:]]
        self.customer_df = pd.DataFrame(customer_data, columns=customer_headers)

    def plot(self):
        plt.figure()
        plt.scatter(self.customer_df['XCOORD.'], self.customer_df['YCOORD.'])
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.scatter(self.customer_df['XCOORD.'].iloc[0], self.customer_df['YCOORD.'].iloc[0], color='red')
        plt.title("map of " + self.name)

