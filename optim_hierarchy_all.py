import sys, os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

def main():
    for i in range(100):
        cmd = 'python optim_hierarchy.py {} --object_id {}'.format(
            'configs/optim_based/abc.yaml', str(i),
        )
        print(cmd)
        # args = cmd.split(" ")
        # process = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    #  universal_newlines=True)
        os.system(cmd)

if __name__=="__main__":
    main()
