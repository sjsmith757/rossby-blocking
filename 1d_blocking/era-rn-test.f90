! include 'mkl_vsl.f90' 
       program MKL_VSL_GAUSSIAN
!        USE MKL_VSL_TYPE 
!        USE MKL_VSL
      integer, parameter :: imax = 2001,mend=600001,mind = 250
      common /array/  aa(imax+1,4),cc(imax+1),a0(imax+1)
      common /brray/  f1(imax+1,3),f2(imax+1,3),f3(imax+1,3)
      common /crray/  d1(imax+1,3),d2(imax+1,3),d3(imax+1,3)
      common /drray/  aad(imax+1),bb(imax+1)
      real(kind=4) r1(1000),r2(1000),del(imax+1),ff(imax+1) ! 
      real(kind=4) s ! 
      real(kind=4) a, b, sigma !
!       TYPE (VSL_STREAM_STATE) :: stream
      integer(kind=4) errcode 
      integer(kind=4) i,j 
      integer brng,method,seed,n
      integer ret,shape,dsadata
      real :: dt,dx,rkappa,pi,alpha
      pi = acos(-1.)
      dt = 60.
      dx = 14000. 
      alpha = 0.55
  !   alpha = 0.4
      beta = 60.
  !   beta = 44.
      c0 = 25.
      tau = 10.*24.*3600.
  !   tau = 12.*24.*3600.
      eps = 16./tau
      n = 1000
      rkappa = 0.1*dx*dx/dt
      sigma = 2.0
!       brng=VSL_BRNG_MT19937 
!     method=VSL_RNG_METHOD_GAUSSIAN_ICDF 
!       method = VSL_RNG_METHOD_UNIFORM_STD
      seed=777
! ***** Initializing ***** 
!       errcode=vslnewstream( stream, brng, seed )


! ***** Initialization *****
      do i = 1,imax+1
        p = 2.*pi*float(i-1)/float(imax-1)
        cc(i) = 34.+7.*cos(2.*p)
        a0(i) = 10.-10.*cos(2.*p)
        bb(i) = sqrt(4.*alpha*cc(i)*20.)+cc(i)
        write(6,*) i,a0(i),cc(i),bb(i)
      enddo

      aa(:,:) = 0.
      f1(:,:) = 0.
      f2(:,:) = 0.
      f3(:,:) = 0.
      d1(:,:) = 0.
      d2(:,:) = 0.
      d3(:,:) = 0.

! ***** Main loop ****
        nfile = 0
      do m = 1,mend 
!       errcode=vsrnggaussian( method, stream, imax+1, ff, 0., sigma ) 
!       errcode=vsrnguniform( method, stream, imax+1, del, 0., eps ) 
        do i = 2,imax
         f1(i,3) = (2.*alpha*a0(i+1)*aa(i+1,3)-2.*alpha*  &
                a0(i-1)*aa(i-1,3))/(2.*dx)
         f2(i,3) = (alpha*aa(i+1,3)**2-alpha*     &
            aa(i-1,3)**2)/(2.*dx)
!        f3(i,3) = -(ff(i+1)*aa(i+1,2)-ff(i-1)*aa(i-1,2))/(2.*dx)
         f3(i,3) = -(beta*aa(i+1,3)-beta*      &
              aa(i-1,3))/(2.*dx)
         d1(i,3) = rkappa*(aa(i+1,1)+aa(i-1,1)-2.*aa(i,1))/(dx*dx)
         d2(i,3) = -aa(i,1)/tau
!        d3(i,3) = del(i)
!        d3(i,3) = eps
         
!        if(m.gt.240000.and.m.le.270000) then
         !if(i.gt.751.and.i.lt.1251) then
          p = 4.*pi*float(i-1201)/2000.
          xc = float(i-1201)
          xcc = -xc*xc/40000.
          tc = float(m-400000)
          tcc = -tc*tc/25000000.
          d3ex = 2.*eps
!         d3(i,3) = eps*(1.+cos(p))
          d3(i,3) = eps
          d3(i,3) = d3(i,3)+d3ex*exp(xcc)*exp(tcc)
!        if(m.gt.400000) d2(i,3) = 0.
       !  endif
       ! endif
         f11 = (23.*f1(i,3)-16.*f1(i,2)+5.*f1(i,1))/12.
         f21 = (23.*f2(i,3)-16.*f2(i,2)+5.*f2(i,1))/12.
         f31 = (23.*f3(i,3)-16.*f3(i,2)+5.*f3(i,1))/12.
         d11 = (23.*d1(i,3)-16.*d1(i,2)+5.*d1(i,1))/12.
         d21 = (23.*d2(i,3)-16.*d2(i,2)+5.*d2(i,1))/12.
         d31 = (23.*d3(i,3)-16.*d3(i,2)+5.*d3(i,1))/12.
         aa(i,4) = aa(i,3)+dt*(f11+f21+f31+d11+d21+d31)
        enddo
        aa(1,4) = aa(imax,4)
        aa(imax+1,4) = aa(2,4)

        a00 = 0.
        do i = 1,imax-1
         a00 = a00 + aa(i,4)/float(imax)
        enddo

        write(6,*) 'End m =',m,aa(501,4),aa(1001,4),a00
  
        aa(:,1) = aa(:,2)
        aa(:,2) = aa(:,3)
        aa(:,3) = aa(:,4)
        f1(:,1) = f1(:,2)
        f1(:,2) = f1(:,3)
        f2(:,1) = f2(:,2)
        f2(:,2) = f2(:,3)
        f3(:,1) = f3(:,2)
        f3(:,2) = f3(:,3)
        d1(:,1) = d1(:,2)
        d1(:,2) = d1(:,3)
        d2(:,1) = d2(:,2)
        d2(:,2) = d2(:,3)
        d3(:,1) = d3(:,2)
        d3(:,2) = d3(:,3)


        if(mod(m,mind).eq.1.and.nfile.le.6000) then
          aad(:) = aa(:,4)
         if(m.ge.374000) then
!           ret = dsadata('aad.df',1,imax+1,aad)
            open(unit=37,file="noboru_out",position='append',status='unknown')
            do i=2,imax
              write(37,*) aad(i)
            enddo
            close(37)
         endif
          nfile = nfile + 1
        endif
      enddo

! ***** Generating ***** 
!     do i = 1,10
!     errcode=vsrnggaussian( method, stream, n, r1, a, sigma ) 
!     errcode=vsrnguniform( method, stream, n, r2, a, b ) 
!     do j = 1, 1000
!     s1 = s1 + r1(j) 
!     s2 = s2 + r2(j) 
!      write(6,*) i,j,r(j)
!     end do
!     end do
!     s1 = s1 / 10000.0
!     s2 = s2 / 10000.0
! ***** Deinitialize ***** errcode=vsldeletestream( stream )
! ***** Printing results *****
!     print *, "Sample mean of normal distribution =",s1,s2

      stop
      end 
