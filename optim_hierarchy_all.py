import sys, os, typing

def start_process_pool(worker_function, parameters: typing.Iterable[typing.Iterable], num_processes, timeout=None):
            # tqdm experiment
            from tqdm import tqdm
            import multiprocessing
            pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=1)
            try:
                # for _ in tqdm(pool.imap_unordered(process_one, obj_list), total=len(obj_list)):
                for _ in tqdm(pool.starmap(worker_function, parameters), total=len(parameters)):
                    pass
                # pool.map_async(process_one, obj_list).get()
            except KeyboardInterrupt:
                # Allow ^C to interrupt from any thread.
                exit()
            pool.close()

            # proven pool
            # with multiprocessing.Pool(processes=num_processes, maxtasksperchild=1) as pool:
            #     results = pool.starmap(worker_function, parameters)
            #     return results

def main():
    # for i in range(100):
    #     cmd = 'python optim_hierarchy.py {} --object_id {}'.format(
    #         'configs/optim_based/abc.yaml', str(i),
    #     )
    #     print(cmd)
    #     # args = cmd.split(" ")
    #     # process = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    #                                 #  universal_newlines=True)
    #     os.system(cmd)
    
    cmds = [('python optim_hierarchy.py {} --object_id {}'.format('configs/optim_based/abc.yaml', str(i)),) for i in range(100)]
    start_process_pool(os.system, cmds, num_processes=25)

if __name__=="__main__":
    main()
