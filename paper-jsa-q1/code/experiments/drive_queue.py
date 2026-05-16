#!/usr/bin/env python3
"""
GNNocRoute-DRL: Drive Queue Controller
========================================
Server-side controller for Colab GPU training via Google Drive.
- Em ghi task file lên Drive
- Colab notebook poll, nhận task, chạy training 
- Colab ghi kết quả về Drive
- Em đọc kết quả

Usage:
  python3 drive_queue.py --train hotspot 300
  python3 drive_queue.py --status
  python3 drive_queue.py --wait
"""

import sys, os, json, time, subprocess, tempfile
from datetime import datetime

RCLONE = 'nasoft-drive:GNNocRoute_Research_Backup/hosoncs/2025/colab-queue'
LOCAL_QUEUE = '/home/opc/.openclaw/workspace/hosoncs/2025/colab-queue'
os.makedirs(LOCAL_QUEUE, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {msg}')

def rclone_cmd(args):
    """Run rclone command."""
    result = subprocess.run(['rclone'] + args, capture_output=True, text=True, timeout=30)
    return result.returncode == 0, result.stdout, result.stderr

def submit_task(task_type, params):
    """Submit a training task to Colab via Drive."""
    task_id = f'task_{int(time.time())}'
    task = {
        'id': task_id,
        'type': task_type,
        'params': params,
        'submitted': datetime.now().isoformat(),
        'status': 'pending'
    }
    
    # Write locally
    local_path = os.path.join(LOCAL_QUEUE, f'{task_id}.task.json')
    with open(local_path, 'w') as f:
        json.dump(task, f, indent=2)
    
    # Upload to Drive
    ok, out, err = rclone_cmd(['copy', local_path, f'{RCLONE}/tasks/'])
    if ok:
        log(f'✅ Task submitted: {task_id} | {task_type} {json.dumps(params)}')
    else:
        log(f'❌ Upload failed: {err}')
    
    return task_id

def check_results(task_id=None):
    """Check for completed results from Colab."""
    # Sync from Drive
    rclone_cmd(['copy', f'{RCLONE}/results/', LOCAL_QUEUE])
    
    if task_id:
        pattern = f'{task_id}.result.json'
    else:
        pattern = '*.result.json'
    
    import glob
    results = []
    for f in sorted(glob.glob(os.path.join(LOCAL_QUEUE, pattern))):
        with open(f) as fp:
            data = json.load(fp)
        results.append(data)
        log(f'📊 Result: {os.path.basename(f)}')
        print(json.dumps(data, indent=2))
    
    return results

def wait_for_result(task_id, timeout_min=30):
    """Wait for a specific task to complete."""
    log(f'⏳ Waiting for {task_id} (timeout: {timeout_min} min)...')
    start = time.time()
    while time.time() - start < timeout_min * 60:
        results = check_results(task_id)
        if results:
            return results[0]
        time.sleep(30)
        log(f'  Still waiting... ({int(time.time()-start)}s)')
    log('⏰ Timeout!')
    return None

def get_status():
    """Get current queue status."""
    # Sync from Drive
    rclone_cmd(['copy', f'{RCLONE}/tasks/', LOCAL_QUEUE])
    rclone_cmd(['copy', f'{RCLONE}/results/', LOCAL_QUEUE])
    rclone_cmd(['copy', f'{RCLONE}/status/', LOCAL_QUEUE])
    
    import glob
    
    tasks = sorted(glob.glob(os.path.join(LOCAL_QUEUE, '*.task.json')))
    results = sorted(glob.glob(os.path.join(LOCAL_QUEUE, '*.result.json')))
    statuses = sorted(glob.glob(os.path.join(LOCAL_QUEUE, 'status_*.json')))
    
    print('=' * 50)
    print('GNNocRoute-DRL: Queue Status')
    print('=' * 50)
    print(f'Tasks pending: {len(tasks)}')
    print(f'Results ready: {len(results)}')
    print(f'Colab statuses: {len(statuses)}')
    
    if statuses:
        with open(statuses[-1]) as f:
            log('Latest Colab status:')
            colab_status = json.load(f)
            print(f'  GPU: {colab_status.get("gpu", "N/A")}')
            print(f'  Runtime: {colab_status.get("runtime", "N/A")}')
            print(f'  Last seen: {colab_status.get("timestamp", "N/A")}')
    
    return {'tasks': len(tasks), 'results': len(results)}


# ===== Training Tasks =====
def submit_training(traffic='hotspot', episodes=300, profile='blackscholes'):
    """Submit a DRL training task to Colab."""
    return submit_task('train_drl', {
        'traffic': traffic,
        'episodes': episodes,
        'profile': profile,
        'inj_rate': 0.1,
    })

def submit_booksim_energy():
    """Submit BookSim2 energy experiments to Colab."""
    return submit_task('booksim_energy', {
        'topologies': ['mesh44', 'mesh88'],
        'algorithms': ['dor', 'adaptive_xy_yx', 'min_adapt'],
        'traffics': ['uniform', 'transpose', 'hotspot'],
        'inj_rates': [0.01, 0.02, 0.05, 0.1],
    })

def submit_parsec_traces():
    """Submit PARSEC benchmark experiments to Colab."""
    return submit_task('parsec_traces', {
        'profiles': ['blackscholes', 'bodytrack', 'fluidanimate', 'canneal'],
        'episodes': 200,
    })


# ===== Main CLI =====
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 drive_queue.py --train <traffic> <episodes>")
        print("  python3 drive_queue.py --energy")
        print("  python3 drive_queue.py --parsec")
        print("  python3 drive_queue.py --status")
        print("  python3 drive_queue.py --wait <task_id>")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == '--train':
        traffic = sys.argv[2] if len(sys.argv) > 2 else 'hotspot'
        episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 300
        task_id = submit_training(traffic, episodes)
        print(f'\nTo check results: python3 drive_queue.py --wait {task_id}')
    
    elif cmd == '--energy':
        task_id = submit_booksim_energy()
        print(f'\nTo check results: python3 drive_queue.py --wait {task_id}')
    
    elif cmd == '--parsec':
        task_id = submit_parsec_traces()
        print(f'\nTo check results: python3 drive_queue.py --wait {task_id}')
    
    elif cmd == '--status':
        get_status()
    
    elif cmd == '--wait':
        if len(sys.argv) > 2:
            result = wait_for_result(sys.argv[2])
            if result:
                print('\n✅ Task completed!')
        else:
            print('Please provide task_id')
    
    elif cmd == '--all':
        """Run all experiments sequentially."""
        log('=== GNNocRoute-DRL: Full Experiment Pipeline ===')
        
        # 1. DRL Training
        log('\n📌 Step 1: DRL Training')
        for traffic in ['hotspot', 'uniform']:
            tid = submit_training(traffic, 300)
            result = wait_for_result(tid, timeout_min=45)
            if result:
                log(f'✅ {traffic} training complete')
        
        # 2. BookSim2 Energy
        log('\n📌 Step 2: BookSim2 Energy')
        tid = submit_booksim_energy()
        result = wait_for_result(tid, timeout_min=60)
        
        # 3. PARSEC
        log('\n📌 Step 3: PARSEC Benchmarks')
        tid = submit_parsec_traces()
        result = wait_for_result(tid, timeout_min=45)
        
        log('\n✅ All experiments submitted to Colab!')
    
    else:
        print(f'Unknown command: {cmd}')
