# Interactive Simulator for Scheduling Algorithms 🚀

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/platform-windows%20%7C%20macos%20%7C%20linux-lightgrey)

An advanced, command-line-based simulator for visualizing and analyzing core scheduling algorithms from Operating Systems (OS) and Electrical & Computer Engineering (ECE). This tool provides a hands-on environment for students, educators, and enthusiasts to understand the trade-offs in efficiency, fairness, and performance inherent in different scheduling strategies.



---

## ## Key Features 💡

* **Multi-Domain Simulation**: Explore algorithms across three distinct domains:
    1.  **Real-Time Scheduling**: Model tasks with hard deadlines.
    2.  **Disk I/O Scheduling**: Simulate disk head movements for data requests.
    3.  **Cognitive Radio Spectrum Allocation**: Manage bandwidth for multiple users.
* **Rich Algorithm Library**: Implements and compares a wide range of classic and advanced algorithms, including **RMS, EDF, LLF, FCFS, SSTF, SCAN, C-SCAN, C-LOOK, Weighted Fair Queuing, and Proportional Fair**.
* **In-depth Performance Analysis**: Goes beyond simple simulation to provide critical metrics:
    * Pre-simulation **schedulability analysis** for real-time systems.
    * **Deadline miss detection** and reporting.
    * Calculation of **total head movement** in disk scheduling.
    * **Jain's Fairness Index** and throughput analysis for spectrum allocation.
* **Insightful Visualizations**: Generates clean, easy-to-understand plots using Matplotlib to visualize timelines, head movement paths, and resource allocation.
* **Interactive CLI**: A user-friendly command-line interface allows you to set custom parameters for every simulation run.

---

## ## Technologies Used

* **Python 3**
* **Matplotlib** for data visualization
* **NumPy** for numerical operations

---

## ## Installation & Setup

Get the simulator up and running in a few simple steps.

1.  **Prerequisites**:
    * Ensure you have **Python 3.8 or newer** installed on your system.

2.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

3.  **Install Dependencies**:
    Install the required Python libraries using pip.
    ```bash
    pip install matplotlib numpy
    ```

---

## ## How to Use

Launch the simulator by running the main Python script from your terminal:

```bash
python advanced_simulator.py
```

You will be greeted by the main menu. Simply follow the on-screen prompts to select a simulator module, enter the required parameters (e.g., number of tasks, head position), and the script will run the simulations and generate the corresponding plots.

**Note**: Each plot is displayed in a separate window. You must **close the current plot window** for the script to proceed to the next simulation or return to the main menu.

---

## ## Simulator Modules

### ### 1. Real-Time Scheduling

Simulates CPU scheduling for tasks with strict timing constraints.

* **Algorithms**:
    * Rate-Monotonic Scheduling (RMS)
    * Earliest Deadline First (EDF)
    * Least Laxity First (LLF)
* **Analysis**:
    * Calculates **CPU Utilization** and performs **schedulability tests** before simulation.
    * Visualizes the schedule timeline and clearly marks **task deadlines** and any **missed deadlines**.



### ### 2. Disk Scheduling

Simulates the movement of a disk's read/write head to service a queue of I/O requests.

* **Algorithms**:
    * First-Come, First-Served (FCFS)
    * Shortest Seek Time First (SSTF)
    * SCAN (Elevator)
    * C-SCAN (Circular SCAN)
    * C-LOOK
* **Analysis**:
    * Calculates and reports the **total head movement** in cylinders for each algorithm.
    * Plots the path of the disk head for a clear visual comparison.



### ### 3. Cognitive Radio Spectrum Allocation

Simulates dynamic and fair allocation of a limited bandwidth spectrum among multiple users.

* **Algorithms**:
    * Weighted Fair Queuing (WFQ)
    * Proportional Fair (PF)
* **Analysis**:
    * Measures the average **throughput** for each user.
    * Calculates **Jain's Fairness Index** to provide a quantitative score of how fairly bandwidth was distributed.
    * Presents results in a bar chart for easy comparison of user allocations.



---

## ## Contributing

Contributions are welcome! If you have suggestions for new algorithms, features, or improvements, please feel free to open an issue or submit a pull request.

## ## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.
