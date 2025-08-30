import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math

# ----------------- Base Class for Simulators -----------------
class Simulator:
    """A base class for our simulators to handle the main loop."""
    def run(self):
        """Abstract method to be implemented by each specific simulator."""
        raise NotImplementedError("Each simulator must implement the 'run' method.")

    def _get_int_input(self, prompt, min_val=1):
        """Helper to get validated integer input from the user."""
        while True:
            try:
                val = int(input(prompt))
                if val >= min_val:
                    return val
                else:
                    print(f"Please enter a value greater than or equal to {min_val}.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

# ----------------- PROJECT 1: Real-Time Scheduling -----------------
class RealTimeSchedulerSimulator(Simulator):
    """
    Simulates and analyzes real-time scheduling algorithms: RMS, EDF, and LLF.
    Includes schedulability analysis and deadline miss detection.
    """
    def run(self):
        n = self._get_int_input("Enter number of tasks: ")
        tasks = []
        for i in range(n):
            exec_time = self._get_int_input(f"Execution time for Task {i+1}: ")
            period = self._get_int_input(f"Period for Task {i+1}: ")
            # In many systems, deadline equals period. We can make them separate for flexibility.
            deadline = self._get_int_input(f"Deadline for Task {i+1} (<= period): ", min_val=exec_time)
            tasks.append({
                "name": f"T{i+1}", "exec": exec_time, "period": period, "deadline": deadline,
                "remaining": 0, "next_deadline": 0
            })

        sim_time = self._get_int_input("Enter simulation time: ", min_val=10)
        
        # Perform Schedulability Analysis before simulation
        self._perform_schedulability_analysis(tasks)

        # Run and plot for each scheduler
        for scheduler_func, name in [
            (self._rms_scheduler, "Rate-Monotonic Scheduling (RMS)"),
            (self._edf_scheduler, "Earliest Deadline First (EDF)"),
            (self._llf_scheduler, "Least Laxity First (LLF)"),
        ]:
            # Reset tasks for each run
            for t in tasks:
                t.update({"remaining": 0, "next_deadline": 0})
            timeline, missed_deadlines = scheduler_func(tasks, sim_time)
            print(f"\n--- {name} Results ---")
            print(f"Timeline: {' '.join(timeline)}")
            if missed_deadlines:
                print(f"🔴 Deadlines MISSED for: {', '.join(sorted(list(set(missed_deadlines))))}")
            else:
                print("🟢 All deadlines met.")
            self._plot_timeline(timeline, tasks, name, missed_deadlines)

    def _perform_schedulability_analysis(self, tasks):
        print("\n--- Schedulability Analysis ---")
        n = len(tasks)
        utilization = sum(t['exec'] / t['period'] for t in tasks)
        print(f"CPU Utilization (U): {utilization:.3f}")

        # EDF Schedulability
        if utilization <= 1:
            print(f"✅ EDF: Schedulable (since U = {utilization:.3f} <= 1)")
        else:
            print(f"❌ EDF: NOT Schedulable (since U = {utilization:.3f} > 1)")
        
        # RMS Schedulability (Liu & Layland bound)
        rms_bound = n * (2**(1/n) - 1)
        if utilization <= rms_bound:
            print(f"✅ RMS: Schedulable (since U = {utilization:.3f} <= {rms_bound:.3f})")
        else:
            print(f"⚠️  RMS: May or may not be schedulable (U = {utilization:.3f} > {rms_bound:.3f}). Test is sufficient but not necessary.")
        print("-" * 30)


    def _run_scheduler(self, tasks, sim_time, priority_key_func):
        timeline = []
        missed_deadlines = []
        
        task_instances = {t['name']: [] for t in tasks}

        for time in range(sim_time):
            # Check for missed deadlines from previous step
            for t in tasks:
                if t['remaining'] > 0 and time >= t['next_deadline']:
                    if t['name'] not in missed_deadlines: # Log only once per instance
                         missed_deadlines.append(t['name'])
                    # Mark the instance that missed
                    if task_instances[t['name']]:
                         task_instances[t['name']][-1]['missed'] = time

            # Release new task instances
            for t in tasks:
                if time % t['period'] == 0:
                    t['remaining'] = t['exec']
                    t['next_deadline'] = time + t['deadline']
                    task_instances[t['name']].append({'arrival': time, 'deadline': t['next_deadline'], 'missed': None})

            runnable = [t for t in tasks if t.get("remaining", 0) > 0]
            
            if runnable:
                # Select task based on the provided priority key function
                task_to_run = priority_key_func(runnable, time)
                task_to_run["remaining"] -= 1
                timeline.append(task_to_run["name"])
            else:
                timeline.append("Idle")
        
        return timeline, missed_deadlines

    def _rms_scheduler(self, tasks, sim_time):
        return self._run_scheduler(tasks, sim_time, lambda runnable, time: min(runnable, key=lambda x: x["period"]))

    def _edf_scheduler(self, tasks, sim_time):
        return self._run_scheduler(tasks, sim_time, lambda runnable, time: min(runnable, key=lambda x: x["next_deadline"]))

    def _llf_scheduler(self, tasks, sim_time):
        def get_laxity(task, current_time):
            return (task['next_deadline'] - current_time) - task['remaining']
        
        def llf_priority(runnable, time):
            return min(runnable, key=lambda t: get_laxity(t, time))
            
        return self._run_scheduler(tasks, sim_time, llf_priority)


    def _plot_timeline(self, result, tasks, title, missed_deadlines):
        fig, ax = plt.subplots(figsize=(15, 3))
        colors = {t["name"]: plt.cm.viridis(i/len(tasks)) for i, t in enumerate(tasks)}
        colors["Idle"] = 'lightgray'

        for i, task_name in enumerate(result):
            ax.barh(0, 1, left=i, color=colors[task_name], edgecolor='black', linewidth=0.5, label=task_name)

        # Plot deadlines and missed deadlines
        task_arrivals = defaultdict(list)
        for i in range(len(result)):
            for t in tasks:
                if i % t['period'] == 0:
                    deadline_time = i + t['deadline']
                    if deadline_time <= len(result):
                        ax.scatter(deadline_time - 0.5, 0, marker='x', color='red', s=100, zorder=3, label=f'{t["name"]} Deadline' if i==0 else "")
        
        # Mark where deadlines were missed
        miss_points = set()
        for t_name in missed_deadlines:
            for t in tasks:
                if t['name'] == t_name:
                    for i in range(len(result)):
                        if i % t['period'] == 0 and i + t['deadline'] < len(result):
                            if result[i + t['deadline']-1] != t['name'] or sum(1 for j in range(i, i+t['deadline']) if result[j] == t['name']) < t['exec']:
                                miss_points.add(i + t['deadline'])
        for point in miss_points:
            ax.axvline(x=point - 0.5, color='red', linestyle='--', linewidth=2, label='Deadline Miss')


        # Create a clean legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.01, 1), loc='upper left')

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_yticks([])
        ax.set_ylim(-0.8, 0.8)
        plt.tight_layout()
        plt.show()

# ----------------- PROJECT 2: Disk Scheduling -----------------
class DiskSchedulerSimulator(Simulator):
    """
    Simulates and visualizes disk scheduling algorithms: FCFS, SSTF, SCAN, C-SCAN, C-LOOK.
    """
    def run(self):
        n = self._get_int_input("Enter number of disk requests: ")
        requests_str = input(f"Enter the {n} requests (space-separated): ")
        requests = [int(r) for r in requests_str.split()]
        head = self._get_int_input("Initial head position: ")
        disk_size = self._get_int_input("Disk size (e.g., 200): ")

        schedulers = {
            "FCFS": self._fcfs,
            "SSTF": self._sstf,
            "SCAN": self._scan,
            "C-SCAN": self._c_scan,
            "C-LOOK": self._c_look
        }

        for name, func in schedulers.items():
            order, movement = func(requests.copy(), head, disk_size)
            print(f"\n--- {name} ---")
            print(f"  Order of service: {order}")
            print(f"  Total head movement: {movement} cylinders")
            self._plot(order, head, f"{name} Disk Scheduling")
            
    def _fcfs(self, req, head, disk_size):
        order = req
        movement = abs(head - order[0]) + sum(abs(order[i] - order[i-1]) for i in range(1, len(order)))
        return order, movement

    def _sstf(self, req, head, disk_size):
        order, movement = [], 0
        current_pos = head
        while req:
            closest = min(req, key=lambda x: abs(x - current_pos))
            movement += abs(closest - current_pos)
            order.append(closest)
            current_pos = closest
            req.remove(closest)
        return order, movement

    def _scan(self, req, head, disk_size, direction='right'):
        order, movement = [], 0
        req.sort()
        
        left = [r for r in req if r < head]
        right = [r for r in req if r >= head]

        if direction == 'right':
            # Move right to the end
            order.extend(sorted(right))
            movement += (disk_size - 1) - head
            # Move left
            order.extend(sorted(left, reverse=True))
            if left: # only add movement if there are requests on the left
                movement += (disk_size - 1) - left[0]
        else: # direction 'left'
            order.extend(sorted(left, reverse=True))
            movement += head
            order.extend(sorted(right))
            if right:
                movement += right[-1]
                
        return order, movement

    def _c_scan(self, req, head, disk_size):
        order, movement = [], 0
        req.sort()
        right = [r for r in req if r >= head]
        left = [r for r in req if r < head]

        # Move right
        order.extend(sorted(right))
        movement += (disk_size - 1) - head
        
        # Jump to beginning and move right again
        if left:
            movement += (disk_size - 1) # Full sweep
            order.extend(sorted(left))
            movement += left[-1] # From 0 to last request

        return order, movement

    def _c_look(self, req, head, disk_size):
        order, movement = [], 0
        req.sort()
        right = [r for r in req if r >= head]
        left = [r for r in req if r < head]

        if not right or not left: # If all requests are on one side, it's just a simple sweep
            order = sorted(req) if (head < req[0] if req else True) else sorted(req, reverse=True)
            movement = abs(head - order[0]) + abs(order[-1] - order[0])
            return order, movement

        # Move right to the last request
        order.extend(sorted(right))
        movement += abs(right[-1] - head)
        
        # Jump to the first request on the left and move right
        movement += abs(right[-1] - left[0])
        order.extend(sorted(left))

        return order, movement

    def _plot(self, order, head, title):
        plt.figure(figsize=(10, 6))
        full_path = [head] + order
        plt.plot(full_path, range(len(full_path)), marker='o', drawstyle='steps-pre')
        plt.scatter(head, 0, c='red', s=100, zorder=5, label='Start Position')
        plt.title(title, fontsize=16)
        plt.ylabel("Sequence Step", fontsize=12)
        plt.xlabel("Cylinder Number", fontsize=12)
        plt.gca().invert_yaxis() # Puts step 0 at the top
        plt.grid(True, linestyle='--')
        plt.legend()
        plt.show()

# ----------------- PROJECT 3: Cognitive Radio -----------------
class CognitiveRadioSimulator(Simulator):
    """
    Simulates cognitive radio spectrum allocation with advanced, fair algorithms.
    """
    def run(self):
        n = self._get_int_input("Enter number of users: ")
        users = []
        for i in range(n):
            bw_demand = self._get_int_input(f"Bandwidth demand for User {i+1} (e.g., in MHz): ")
            priority = self._get_int_input(f"Priority for User {i+1} (1=highest): ")
            users.append({"name": f"U{i+1}", "demand": bw_demand, "priority": priority, "avg_throughput": 0.01}) # small initial throughput

        total_bw = self._get_int_input("Total available bandwidth: ")
        time = self._get_int_input("Simulation time (steps): ")

        # Run simulations
        wfq_allocations = self._weighted_fair_queuing(users, total_bw, time)
        pf_allocations = self._proportional_fair(users, total_bw, time)
        
        # Analyze and plot results
        self._analyze_and_plot("Weighted Fair Queuing", wfq_allocations, users)
        self._analyze_and_plot("Proportional Fair", pf_allocations, users)

    def _jains_fairness_index(self, throughputs):
        if not throughputs: return 0
        sum_x = sum(throughputs)
        sum_x2 = sum(x**2 for x in throughputs)
        if sum_x2 == 0: return 1.0 # Perfect fairness if all are zero
        return (sum_x**2) / (len(throughputs) * sum_x2)

    def _analyze_and_plot(self, title, allocations, users):
        user_names = [u['name'] for u in users]
        total_allocations = [sum(allocations[t][u['name']] for t in range(len(allocations))) for u in users]
        avg_throughput = [t / len(allocations) for t in total_allocations]
        
        fairness = self._jains_fairness_index(avg_throughput)
        
        print(f"\n--- {title} Results ---")
        for i, user in enumerate(users):
            print(f"  {user['name']}: Average Throughput = {avg_throughput[i]:.2f} MHz")
        print(f"  Jain's Fairness Index: {fairness:.4f} (1.0 is perfect fairness)")

        plt.figure(figsize=(10, 6))
        plt.bar(user_names, avg_throughput, color=plt.cm.tab20(np.arange(len(users))))
        plt.title(f"{title} - Average Throughput per User", fontsize=16)
        plt.xlabel("User", fontsize=12)
        plt.ylabel("Average Bandwidth (MHz)", fontsize=12)
        plt.show()

    def _weighted_fair_queuing(self, users, total_bw, time):
        allocations = []
        # Weights are inverse of priority (lower priority number = higher weight)
        weights = {u['name']: 1.0/u['priority'] for u in users}
        total_weight = sum(weights.values())

        for _ in range(time):
            time_slot_alloc = defaultdict(int)
            for user in users:
                share = (weights[user['name']] / total_weight) * total_bw
                # Allocate the smaller of their demand or their fair share
                time_slot_alloc[user['name']] = min(user['demand'], share)
            allocations.append(time_slot_alloc)
        return allocations
        
    def _proportional_fair(self, users, total_bw, time):
        allocations = []
        # Initialize average throughput for each user
        avg_throughput = {u['name']: 0.01 for u in users} # Start with a small non-zero value
        alpha = 0.1 # Smoothing factor for updating average throughput

        for t in range(time):
            # Simulate varying channel quality (random factor for simplicity)
            channel_quality = {u['name']: np.random.uniform(0.5, 1.5) for u in users}
            
            # Calculate PF metric for each user
            metrics = {}
            for u in users:
                # Potential rate is demand modulated by channel quality
                potential_rate = u['demand'] * channel_quality[u['name']]
                metrics[u['name']] = potential_rate / avg_throughput[u['name']]
            
            # Sort users by their PF metric in descending order
            sorted_users = sorted(users, key=lambda u: metrics[u['name']], reverse=True)
            
            # Allocate bandwidth
            bw_remaining = total_bw
            time_slot_alloc = defaultdict(int)
            for user in sorted_users:
                if bw_remaining > 0:
                    alloc = min(user['demand'], bw_remaining)
                    time_slot_alloc[user['name']] = alloc
                    bw_remaining -= alloc
            allocations.append(time_slot_alloc)

            # Update average throughput using an exponential moving average
            for user in users:
                current_alloc = time_slot_alloc[user['name']]
                avg_throughput[user['name']] = (1 - alpha) * avg_throughput[user['name']] + alpha * current_alloc
        
        return allocations

# ----------------- MAIN MENU -----------------
def main():
    simulators = {
        "1": RealTimeSchedulerSimulator,
        "2": DiskSchedulerSimulator,
        "3": CognitiveRadioSimulator
    }

    while True:
        print("\n" + "="*15 + " OS-ECE Advanced Simulator " + "="*15)
        print("1. Real-Time Scheduling (RMS, EDF, LLF) with Analysis")
        print("2. Disk Scheduling (FCFS, SSTF, SCAN, C-SCAN, C-LOOK)")
        print("3. Cognitive Radio Spectrum Allocation (WFQ, Proportional Fair)")
        print("4. Exit")
        print("="*54)

        choice = input("Enter choice: ")
        if choice in simulators:
            try:
                simulator = simulators[choice]()
                simulator.run()
            except Exception as e:
                print(f"An error occurred: {e}")
        elif choice == "4":
            print("Exiting... Goodbye! 👋")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()