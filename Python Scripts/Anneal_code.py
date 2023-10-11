
        # mass = 1.03499e-4* np.array([183.84, 10.00784, 4.002602])

        # N_anneal = int(5)

        # t_anneal = int(1e2)

        # E0 = 1

        # EN = 0.1

        # constant = (2/(3*8.617333262e-5))

        # T0 = constant*E0

        # TN = constant*EN

        # decay_constant = (1/N_anneal)*np.log(T0/TN)

        # for i in range(N_anneal):
            
        #     rng_num = np.random.randint(low = 0, high = 10000)

        #     for element in range(len(xyz_inter)):
                
        #         lmp.command('group int_%d type %d' % (element+1, element+1))

        #         if len(xyz_inter[element]) > 1:

        #             T = T0*np.exp(-decay_constant*i)

        #             lmp.command('velocity int_%d create %f %d dist gaussian mom yes rot no units box' % (element+1 ,T, rng_num) )
                
        #         elif len(xyz_inter[element]) == 1:

        #             E = E0*np.exp(-decay_constant*i)

        #             speed = np.sqrt(2*E/mass[element])

        #             unit_vel = np.hstack( [np.random.randn(1), np.random.randn(1), np.random.randn(1)] )

        #             unit_vel /= np.linalg.norm(unit_vel)

        #             vel = speed*unit_vel

        #             lmp.command('velocity int_%d set %f %f %f sum yes units box' % (element+1,vel[0], vel[1], vel[2]))
                
        #         lmp.command('run 0')

        #         lmp.command('timestep %f' % 2e-3)

        #         lmp.command('thermo 50')

        #         lmp.command('thermo_style custom step temp pe press') 

        #         lmp.command('run %d' % t_anneal)
                
        #         lmp.command('minimize 1e-5 1e-8 10 10')

        #         lmp.command('minimize 1e-5 1e-8 10 100')
                
        #         #lmp.command('fix free all box/relax aniso 0.0')

        #         lmp.command('minimize 1e-5 1e-8 100 1000')

        #         lmp.command('run 0')
