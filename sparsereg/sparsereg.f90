      MODULE SPARSEREG
!
!     Determine double precision and set constants.
!
      IMPLICIT NONE
      INTEGER , PARAMETER :: DBLE_PREC = KIND (0.0D0)
      REAL (KIND=DBLE_PREC), PARAMETER :: ZERO  = 0.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: ONE   = 1.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: TWO   = 2.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: THREE = 3.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: FOUR  = 4.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: FIVE  = 5.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: SIX   = 6.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: SEVEN = 7.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: EIGHT = 8.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: NINE  = 9.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: TEN   = 10.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: HALF  = ONE/TWO
      REAL (KIND=DBLE_PREC), PARAMETER :: PI    = 3.14159265358979_DBLE_PREC
!
      CONTAINS
!
      SUBROUTINE PENALTY_FUN(BETA,RHO,ETA,PENTYPE,PEN,D1PEN,D2PEN,DPENDRHO)
!
!     This subroutine calculates the penalty value, first derivative, 
!     and second derivatives of the ENET, LOG, MCP, POWER, or SCAD penalties
!     with index parameter ETA
!       ENET: rho*((eta-1)*beta**2/2+(2-eta)*abs(beta)), 1<=eta<=2
!       LOG: rho*log(eta+abs(beta)), eta>0 (eta=0 implies continuous LOG penalty)
!       MCP: eta>0
!       POWER: rho*abs(beta)**eta, 0<eta<=2
!       SCAD: eta>2
!
      IMPLICIT NONE
      CHARACTER(LEN=*), INTENT(IN) :: PENTYPE
      LOGICAL :: CONTLOG
      REAL(KIND=DBLE_PREC), PARAMETER :: EPS=1E-8
      REAL(KIND=DBLE_PREC) :: ETA,RHO
      REAL(KIND=DBLE_PREC), DIMENSION(:) :: BETA,PEN
      REAL(KIND=DBLE_PREC), DIMENSION(SIZE(BETA)) :: ABSBETA
      REAL(KIND=DBLE_PREC), OPTIONAL, DIMENSION(:) :: D1PEN,D2PEN,DPENDRHO
!
!     Check nonnegativity of tuning parameter
!
      IF (RHO<ZERO) THEN
         PRINT*,"THE TUNING PARAMETER MUST BE NONNEGATIVE."
         RETURN
      END IF
!
!     Penalty values
!
      ABSBETA = ABS(BETA)
      SELECT CASE(PENTYPE)
      CASE("ENET")
         IF (ETA<ONE.OR.ETA>TWO) THEN
            PRINT*,"THE ENET PARAMETER ETA SHOULD BE IN [1,2]."
            RETURN
         END IF
         PEN = RHO*(HALF*(ETA-ONE)*BETA*BETA+(TWO-ETA)*ABSBETA)
      CASE("LOG")
         IF (ETA==ZERO) THEN
            ETA = SQRT(RHO)
            CONTLOG = .TRUE.
         ELSEIF (ETA<ZERO) THEN
            PRINT*,"THE LOG PENALTY PARAMETER ETA SHOULD BE NONNEGATIVE."
            RETURN
         END IF
         PEN = RHO*LOG(ETA+ABSBETA)      
      CASE("MCP")
         IF (ETA<=ZERO) THEN
            PRINT*,"THE MCP PARAMETER ETA SHOULD BE POSITIVE."
            RETURN
         END IF
         WHERE (ABSBETA<RHO*ETA)
            PEN = RHO*ABSBETA - HALF*BETA*BETA/ETA
         ELSEWHERE
            PEN = HALF*RHO*RHO*ETA
         END WHERE
      CASE("POWER")
         IF (ETA<=ZERO.OR.ETA>TWO) THEN
            PRINT*,"THE EXPONENT PARAMETER ETA SHOULD BE IN (0,2]."
            RETURN
         END IF
         PEN = RHO*ABSBETA**ETA
      CASE("SCAD")
         IF (ETA<=TWO) THEN
            PRINT*,"THE SCAD PARAMETER ETA SHOULD BE GREATER THAN 2."
            RETURN
         END IF
         WHERE (ABSBETA<RHO)
            PEN = RHO*ABSBETA
         ELSEWHERE (ABSBETA<ETA*RHO)
            PEN = RHO*RHO + ETA*RHO*(ABSBETA-RHO)/(ETA-ONE) &
               - HALF*(BETA*BETA-RHO*RHO)/(ETA-ONE)
         ELSEWHERE
            PEN = HALF*RHO*RHO*(ETA+ONE)
         END WHERE
      END SELECT
!
!     First derivative of penalty function
!
      IF (PRESENT(D1PEN)) THEN
         SELECT CASE(PENTYPE)
         CASE("ENET")
            D1PEN = RHO*((ETA-ONE)*ABSBETA+TWO-ETA)
         CASE("LOG")
            D1PEN = RHO/(ETA+ABSBETA)
         CASE("MCP")
            WHERE (ABSBETA<RHO*ETA)
               D1PEN = RHO - ABSBETA/ETA
            ELSEWHERE
               D1PEN = ZERO
            END WHERE
         CASE("POWER")
            WHERE (ABSBETA<EPS)
               D1PEN = RHO*ETA*EPS**(ETA-ONE)
            ELSEWHERE
               D1PEN = RHO*ETA*ABSBETA**(ETA-ONE)
            END WHERE               
         CASE("SCAD")
            WHERE (ABSBETA<RHO)
               D1PEN = RHO
            ELSEWHERE (ABSBETA<ETA*RHO)
               D1PEN = (ETA*RHO-ABSBETA)/(ETA-ONE)
            ELSEWHERE
               D1PEN = ZERO
            END WHERE         
         END SELECT
      END IF
!
!     Second derivative of penalty function
!
      IF (PRESENT(D2PEN)) THEN
         SELECT CASE(PENTYPE)
         CASE("ENET")
            D2PEN = RHO*(ETA-ONE)
         CASE("LOG")
            D2PEN = -D1PEN/(ETA+ABSBETA)
         CASE("MCP")
            WHERE (ABSBETA<RHO*ETA)
               D2PEN = -ONE/ETA
            ELSEWHERE
               D2PEN = ZERO
            END WHERE
         CASE("POWER")
            WHERE (ABSBETA<EPS)
               D2PEN = RHO*ETA*(ETA-ONE)*EPS**(ETA-TWO)
            ELSEWHERE
               D2PEN = RHO*ETA*(ETA-ONE)*ABSBETA**(ETA-TWO)
            END WHERE           
         CASE("SCAD")
            WHERE (ABSBETA<RHO.OR.ABSBETA>ETA*RHO)
               D2PEN = ZERO
            ELSEWHERE
               D2PEN = ONE/(ONE-ETA)
            END WHERE         
         END SELECT
      END IF
!
!     Second mixed derivative of penalty function
!
      IF (PRESENT(DPENDRHO)) THEN
         SELECT CASE(PENTYPE)
         CASE("ENET")
            DPENDRHO = (ETA-ONE)*ABSBETA+TWO-ETA
         CASE("LOG")
            DPENDRHO = ONE/(ETA+ABSBETA)
            IF (CONTLOG) THEN
               DPENDRHO = DPENDRHO*(ONE-HALF*ETA*DPENDRHO)
            END IF         
         CASE("MCP")
            WHERE (ABSBETA<RHO*ETA)
               DPENDRHO = ONE
            ELSEWHERE
               DPENDRHO = ZERO
            END WHERE
         CASE("POWER")
            WHERE (ABSBETA<EPS)
               DPENDRHO = ETA*EPS**(ETA-ONE)
            ELSEWHERE
               DPENDRHO = ETA*ABSBETA**(ETA-ONE)
            END WHERE         
         CASE("SCAD")
            WHERE (ABSBETA<RHO)
               DPENDRHO = ONE
            ELSEWHERE (ABSBETA<ETA*RHO)
               DPENDRHO = ETA/(ETA-ONE)
            ELSEWHERE
               DPENDRHO = ZERO
            END WHERE         
         END SELECT
      END IF
      END SUBROUTINE PENALTY_FUN
!
      FUNCTION LSQ_THRESHOLDING(A,B,RHO,ETA,PENTYPE) RESULT(XMIN)
!
!     This subroutine performs univariate soft thresholding for least squares:
!        argmin .5*a*x^2+b*x+PEN(abs(x),eta).
!
      IMPLICIT NONE
      CHARACTER(LEN=*), INTENT(IN) :: PENTYPE      
      LOGICAL :: DOBISECTION
      REAL(KIND=DBLE_PREC), PARAMETER :: EPS=1E-8
      REAL(KIND=DBLE_PREC) :: A,ABSB,B,DL,DM,DR,ETA,F1,F2,FXMIN,FXMIN2
      REAL(KIND=DBLE_PREC) :: RHO,XL,XM,XMIN,XMIN2,XR
!
!     Check tuning parameter
!
      IF (RHO<ZERO) THEN
         PRINT*, "PENALTY TUNING CONSTANT SHOULD BE NONNEGATIVE"
         RETURN
      END IF
      
!
!     Transform to format 0.5*a*(x-b)^2
!
      IF (A<=ZERO) THEN
         PRINT*, "QUADRATIC COEFFICIENT A MUST BE POSITIVE"
         RETURN
      END IF
      B = -B/A
      ABSB = ABS(B)
!
!     Thresholding
!
      IF (RHO<EPS) THEN
         XMIN = B
         RETURN
      ELSE
         SELECT CASE(PENTYPE)
         CASE("ENET")
            IF (ETA<ONE.OR.ETA>TWO) THEN
               PRINT*,"THE ENET PARAMETER ETA SHOULD BE IN [1,2]."
               RETURN
            ELSEIF (ABS(ETA-TWO)<EPS) THEN
               XMIN = A*B/(A+RHO)
               RETURN
            END IF
            XMIN = A*B-RHO*(TWO-ETA)
            IF (XMIN>ZERO) THEN
               XMIN = XMIN/(A+RHO*(ETA-1))
               RETURN
            END IF
            XMIN = A*B+RHO*(TWO-ETA)
            IF (XMIN<ZERO) THEN
               XMIN = XMIN/(A+RHO*(ETA-1))
               RETURN
            END IF
            XMIN = ZERO
         CASE("LOG")
            IF (ETA<ZERO) THEN
               PRINT *, "PARAMETER ETA FOR LOG PENALTY SHOULD BE POSITIVE"
               RETURN
            ELSE IF (ETA==ZERO) THEN
               ETA = SQRT(RHO)
            END IF
            IF (RHO<=A*ABSB*ETA) THEN
               XMIN = SIGN(HALF*(ABSB-ETA+ &
                  SQRT((ABSB+ETA)*(ABSB+ETA)-FOUR*RHO/A)),B)            
            ELSEIF (RHO<=A*ETA*ETA) THEN
               XMIN = ZERO
            ELSEIF (RHO>=A*(ETA+ABSB)*(ETA+ABSB)/FOUR) THEN
               XMIN = ZERO
            ELSE
               XMIN = SIGN(HALF*(ABSB-ETA+ &
                  SQRT((ABSB+ETA)*(ABSB+ETA)-FOUR*RHO/A)),B)
               F1 = HALF*A*B*B+RHO*LOG(ETA)
               F2 = HALF*A*(XMIN-B)*(XMIN-B)+RHO*LOG(ETA+ABS(XMIN))
               IF (F1<F2) THEN
                  XMIN = ZERO
               END IF
            END IF
            XMIN = SIGN(XMIN,B)
         CASE("MCP")
            IF (ETA<=ZERO) THEN
               PRINT*,"THE MCP PARAMETER ETA SHOULD BE POSITIVE."
               RETURN
            END IF
            IF (RHO<=A*ABSB) THEN
               IF (ABSB<=RHO*ETA) THEN
                  XMIN = ETA*(A*ABSB-RHO)/(A*ETA-ONE)
               ELSE
                  XMIN = ABSB
               END IF
            ELSEIF (A*ETA>=ONE) THEN
               XMIN = ZERO
            ELSEIF (RHO*ETA>=ABSB) THEN
               XMIN = ZERO
            ELSE
               IF (A*B*B<RHO*RHO*ETA) THEN
                  XMIN = ZERO
               ELSE
                  XMIN = ABSB
               END IF
            END IF
            XMIN = SIGN(XMIN,B)
         CASE("POWER")
            IF (ETA<=ZERO.OR.ETA>TWO) THEN
               PRINT*,"THE EXPONENT ETA SHOULD BE IN (0,2]."
               RETURN
            ELSEIF (ABS(ETA-TWO)<EPS) THEN
               XMIN = A*B/(A+TWO*RHO)
               RETURN
            ELSEIF (ABS(ETA-ONE)<EPS) THEN
               IF (B-RHO/A>ZERO) THEN
                  XMIN = B-RHO/A
               ELSEIF (B+RHO/A<ZERO) THEN
                  XMIN = B+RHO/A
               ELSE
                  XMIN = ZERO
               END IF
               RETURN
            END IF
            DOBISECTION = .FALSE.
            IF (ETA>ONE) THEN
               XL = ZERO
               DL = -A*ABSB
               DOBISECTION = .TRUE.
            ELSEIF (A+RHO*ETA*(ETA-ONE)*ABSB**(ETA-TWO)<=ZERO) THEN
               XMIN = ZERO
            ELSE
               XL = (A/RHO/ETA/(ONE-ETA))**(ONE/(ETA-TWO))
               DL = A*(XL-ABSB)+RHO*ETA*XL**(ETA-ONE)
               IF (DL>=ZERO) THEN
                  XMIN = ZERO
               ELSE
                  DOBISECTION = .TRUE.
               END IF
            END IF
            IF (DOBISECTION) THEN
               XR = ABSB
               DR = RHO*ETA*XR**(ETA-ONE)
               DO
                  XM = HALF*(XL+XR)
                  DM = A*(XM-ABSB)+RHO*ETA*XM**(ETA-ONE)
                  IF (DM>EPS) THEN
                     XR = XM
                     DR = DM
                  ELSEIF (DM<-EPS) THEN
                     XL = XM
                     DL = DM
                  ELSE
                     XMIN = XM
                     EXIT
                  END IF
                  IF (ABS(XL-XR)<EPS) THEN
                     XMIN = XM
                     EXIT
                  END IF
               END DO
               IF ((ETA<ONE) .AND. &
                  (HALF*A*(XMIN-ABSB)**2+RHO*ABS(XMIN)**ETA>HALF*A*ABSB**2)) THEN
                  XMIN = ZERO
               END IF
            END IF
            XMIN = SIGN(XMIN,B)
         CASE("SCAD")
            IF (ETA<=TWO) THEN
               PRINT*,"THE SCAD PARAMETER ETA SHOULD BE GREATER THAN 2."
               RETURN
            END IF 
            IF (RHO<=A*ABSB) THEN
               IF (A*(RHO-ABSB)+RHO>=ZERO) THEN
                  XMIN = ABSB - RHO/A
               ELSEIF (RHO*ETA-ABSB>=ZERO) THEN
                  XMIN = (A*ABSB*(ETA-ONE)-RHO*ETA)/(A*(ETA-ONE)-ONE)
               ELSE
                  XMIN = ABSB
               END IF
            ELSEIF (ABSB<=RHO*ETA.OR.A*(ETA-ONE)>=ONE) THEN
               XMIN = ZERO
            ELSE
               IF (A*B*B<RHO*RHO*(ETA+ONE)) THEN
                  XMIN = ZERO
               ELSE
                  XMIN = ABSB
               END IF
            END IF
            XMIN = SIGN(XMIN,B)
!            XMIN = ZERO
!            IF (RHO>=A*ABS(B)) RETURN
!            FXMIN = HALF*A*B*B
!            XMIN2 = B-SIGN(RHO,B)/A
!            FXMIN2 = HALF*A*(XMIN2-B)*(XMIN2-B)+RHO*ABS(XMIN2)
!            IF (FXMIN2<FXMIN) THEN
!               XMIN = XMIN2
!               FXMIN = FXMIN2
!            END IF
!            IF (RHO>A/(A+ONE)*ABS(B)) RETURN
!            IF (RHO<ABS(B)/ETA) THEN
!               XMIN2 = B
!               FXMIN2 = HALF*A*(XMIN2-B)*(XMIN2-B) + HALF*RHO*RHO*(ETA+ONE)
!            ELSE
!               XMIN2 = (A*B*(ETA-1)-ETA*SIGN(RHO,B))/(A*(ETA-ONE)-ONE)
!               FXMIN2 = HALF*A*(XMIN2-B)*(XMIN2-B) + RHO*RHO &
!                  + ETA/(ETA-ONE)*RHO*(ABS(XMIN2)-RHO) &
!                  - HALF*(XMIN2*XMIN2-RHO*RHO)/(ETA-ONE)
!            END IF
!            IF (FXMIN2<FXMIN) THEN
!               XMIN = XMIN2
!            END IF                    
         END SELECT
      END IF
      END FUNCTION LSQ_THRESHOLDING
!
      FUNCTION MAX_RHO(A,B,PENTYPE,PENPARAM) RESULT(MAXRHO)
!
!     This subroutine finds the maximum penalty constant rho such that
!     argmin 0.5*A*x^2+B*x+penalty(x,rho) is nonzero. Current options for
!     PENTYPE are "ENET","LOG","MCP","POWER","SCAD". PENPARAM contains the
!     optional parameter for the penalty function.
!
      CHARACTER(LEN=*), INTENT(IN) :: PENTYPE
      REAL(KIND=DBLE_PREC) :: A,B,L,M,MAXRHO,R,ROOTL,ROOTM,ROOTR
      REAL(KIND=DBLE_PREC), PARAMETER :: EPS=1E-8
      REAL(KIND=DBLE_PREC), DIMENSION(:) :: PENPARAM
!
!     Locate the max rho
!
      SELECT CASE(PENTYPE)
      CASE("ENET")
         IF (PENPARAM(1)==TWO) THEN
            MAXRHO = TEN*A
         ELSE
            MAXRHO = ABS(B)/(TWO-PENPARAM(1))
         END IF
      CASE("LOG")
         IF (PENPARAM(1)==ZERO) THEN
            IF (A<=ONE) THEN
               MAXRHO = B*B
               RETURN
            ELSE
               L = B*B
               R = TWO*L
               DO WHILE(LSQ_THRESHOLDING(A,B,R,PENPARAM(1),"LOG")>ZERO)
                  L = R
                  R = TWO*R
               END DO
            END IF
         ELSE
            L = ABS(B*PENPARAM(1))
            R = A*(PENPARAM(1)+ABS(B)/A)**2/FOUR
         END IF
         ROOTL = LSQ_THRESHOLDING(A,B,L,PENPARAM(1),"LOG")
         ROOTR = LSQ_THRESHOLDING(A,B,R,PENPARAM(1),"LOG")
         DO
            M = HALF*(L+R)
            ROOTM = LSQ_THRESHOLDING(A,B,M,PENPARAM(1),"LOG")
            IF (ROOTM==ZERO) THEN
               R = M
               ROOTR = ROOTM
            ELSE
               L = M
               ROOTL = ROOTM
            END IF
            IF (ABS(R-L)<EPS) THEN
               MAXRHO = M
               EXIT
            END IF
         END DO
         RETURN                  
      CASE("MCP")
         L = ZERO
         R = ABS(B)/PENPARAM(1)
         ROOTL = LSQ_THRESHOLDING(A,B,L,PENPARAM(1),"MCP")
         ROOTR = LSQ_THRESHOLDING(A,B,R,PENPARAM(1),"MCP")
         DO
            M = HALF*(L+R)
            ROOTM = LSQ_THRESHOLDING(A,B,M,PENPARAM(1),"MCP")
            IF (ROOTM==ZERO) THEN
               R = M
               ROOTR = ROOTM
            ELSE
               L = M
               ROOTL = ROOTM
            END IF
            IF (ABS(R-L)<EPS) THEN
               MAXRHO = M
               EXIT
            END IF
         END DO
      CASE("POWER")
         IF (PENPARAM(1)==ONE) THEN
            MAXRHO = ABS(B)
            RETURN
         ELSEIF (PENPARAM(1)<ONE) THEN
            L = ZERO
            ROOTL = LSQ_THRESHOLDING(A,B,L,PENPARAM(1),"POWER")
            R = ONE
            ROOTR = LSQ_THRESHOLDING(A,B,R,PENPARAM(1),"POWER")
            DO WHILE(ROOTR>ZERO)
               L = R
               ROOTL = ROOTR
               R = TWO*R
               ROOTR = LSQ_THRESHOLDING(A,B,R,PENPARAM(1),"POWER")
            END DO
            DO
               M = HALF*(L+R)
               ROOTM = LSQ_THRESHOLDING(A,B,M,PENPARAM(1),"POWER")
               IF (ROOTM==ZERO) THEN
                  R = M
                  ROOTR = ROOTM
               ELSE
                  L = M
                  ROOTL = ROOTM
               END IF
               IF (ABS(R-L)<EPS) THEN
                  MAXRHO = M
                  EXIT
               END IF
            END DO
            RETURN                  
         ELSEIF (PENPARAM(1)>ONE) THEN
            R = ONE
            ROOTR = LSQ_THRESHOLDING(A,B,R,PENPARAM(1),"POWER")
            DO WHILE(ABS(ROOTR)>ABS(B)/A/TEN)
               R = TWO*R
               ROOTR = LSQ_THRESHOLDING(A,B,R,PENPARAM(1),"POWER")
            END DO
            MAXRHO = R
            RETURN
         END IF
      CASE("SCAD")
         B = -B/A
         L = A/(A+ONE)*ABS(B)
         R = A*ABS(B)
         ROOTL = LSQ_THRESHOLDING(A,B,L,PENPARAM(1),"SCAD")
         ROOTR = LSQ_THRESHOLDING(A,B,R,PENPARAM(1),"SCAD")
         DO
            M = HALF*(L+R)
            ROOTM = LSQ_THRESHOLDING(A,B,M,PENPARAM(1),"SCAD")
            IF (ROOTM==ZERO) THEN
               R = M
               ROOTR = ROOTM
            ELSE
               L = M
               ROOTL = ROOTM
            END IF
            IF (ABS(R-L)<EPS) THEN
               MAXRHO = M
               EXIT
            END IF
         END DO
         RETURN
      END SELECT
      END FUNCTION MAX_RHO
!
      SUBROUTINE PENALIZED_L2_REGRESSION(ESTIMATE,X,Y,WT,LAMBDA, &
         SUM_X_SQUARES,PENIDX,MAXITERS,PENTYPE,PENPARAM)
!
!     This subroutine carries out penalized L2 regression with design
!     matrix X, dependent variable Y, weights W and penalty constant 
!     LAMBDA. Note that the rows of X correspond to cases and the columns
!     to predictors.  The SUM_X_SQUARES should have entries SUM(W*X(:,I)**2)
!
      IMPLICIT NONE
      CHARACTER(LEN=*), INTENT(IN) :: PENTYPE
      INTEGER :: I,ITERATION,M,MAXITERS,N
      REAL(KIND=DBLE_PREC) :: CRITERION=1E-4,EPS=1E-8
      REAL(KIND=DBLE_PREC) :: A,B,LAMBDA,NEW_OBJECTIVE,OLDROOT
      REAL(KIND=DBLE_PREC) :: OBJECTIVE,ROOTDIFF
      LOGICAL, DIMENSION(:) :: PENIDX
      REAL(KIND=DBLE_PREC), DIMENSION(:) :: ESTIMATE,PENPARAM,SUM_X_SQUARES,WT,Y
      REAL(KIND=DBLE_PREC), DIMENSION(:,:) :: X
      LOGICAL, DIMENSION(SIZE(ESTIMATE)) :: NZIDX
      REAL(KIND=DBLE_PREC), DIMENSION(SIZE(Y)) :: R
      REAL(KIND=DBLE_PREC), DIMENSION(SIZE(ESTIMATE)) :: PENALTY
!
!     Check that the number of cases is well defined.
!
      IF (SIZE(Y)/=SIZE(X,1)) THEN
         PRINT*," THE NUMBER OF CASES IS NOT WELL DEFINED."
         RETURN
      END IF
!
!     Check that the number of predictors is well defined.
!
      IF (SIZE(ESTIMATE)/=SIZE(X,2)) THEN
         PRINT*, " THE NUMBER OF PREDICTORS IS NOT WELL DEFINED."
         RETURN
      END IF
!
!     Check the index for penalized predictors
!
      IF (SIZE(PENIDX)/=SIZE(X,2)) THEN
         PRINT*, " THE PENALTY INDEX ARRAY IS NOT WELL DEFINED"
         RETURN
      END IF
!
!     Initialize the number of cases M and the number of regression
!     coefficients N.
!
      M = SIZE(Y)
      N = SIZE(ESTIMATE)
!
!     Initialize the residual vector R, the PENALTY, the loss L2, and the
!     objective function.
!
      IF (ANY(ABS(ESTIMATE)>EPS)) THEN
         R = Y-MATMUL(X,ESTIMATE)
      ELSE
         R = Y
      END IF
      CALL PENALTY_FUN(ESTIMATE,LAMBDA,PENPARAM(1),PENTYPE,PENALTY)
      OBJECTIVE = HALF*SUM(WT*R*R)+SUM(PENALTY,PENIDX)
      PRINT*, "OBJECTIVE = "
      PRINT*, OBJECTIVE
!
!     Initialize maximum number of iterations
!
      IF (MAXITERS<=0) THEN
         MAXITERS = 1000
      END IF
!
!     Enter the main iteration loop.
!
      DO ITERATION = 1,MAXITERS
         DO I = 1,N
            OLDROOT = ESTIMATE(I)           
            A = SUM_X_SQUARES(I)
            B = - SUM(WT*R*X(:,I)) - A*OLDROOT
            IF (PENIDX(I)) THEN
               ESTIMATE(I) = LSQ_THRESHOLDING(A,B,LAMBDA,PENPARAM(1),PENTYPE)
            ELSE
               ESTIMATE(I) = -B/A
            END IF
            ROOTDIFF = ESTIMATE(I)-OLDROOT
            IF (ABS(ROOTDIFF)>EPS) THEN
               R = R - ROOTDIFF*X(:,I)
            END IF
         END DO
!
!     Output the iteration number and value of the objective function.
!
         CALL PENALTY_FUN(ESTIMATE,LAMBDA,PENPARAM(1),PENTYPE,PENALTY)
         NEW_OBJECTIVE = HALF*SUM(WT*R*R)+SUM(PENALTY,PENIDX)
         IF (ITERATION==1.OR.MOD(ITERATION,1)==0) THEN
            PRINT*," ITERATION = ",ITERATION," FUN = ",NEW_OBJECTIVE
         END IF
!
!     Check for a descent failure or convergence.  If neither occurs,
!     record the new value of the objective function.
!
         IF (NEW_OBJECTIVE>OBJECTIVE+EPS) THEN
            PRINT*," *** ERROR *** OBJECTIVE FUNCTION INCREASE AT ITERATION",ITERATION
            RETURN
         END IF
         IF ((OBJECTIVE-NEW_OBJECTIVE)<CRITERION*(ABS(OBJECTIVE)+ONE)) THEN
            RETURN
         ELSE
            OBJECTIVE = NEW_OBJECTIVE
         END IF
      END DO
      END SUBROUTINE PENALIZED_L2_REGRESSION
!
      END MODULE SPARSEREG

      PROGRAM TEST
      USE SPARSEREG
      IMPLICIT NONE
      INTEGER, PARAMETER :: N=100, P=5
      INTEGER :: MAXITERS
      LOGICAL, DIMENSION(P) :: PENIDX
      REAL(KIND=DBLE_PREC) :: A,B,LAMBDA,RHO=ONE,ETA=ZERO,XMIN
      REAL(KIND=DBLE_PREC), DIMENSION(P) :: BETA=(/ONE,TWO,THREE,FOUR,FIVE/)
      REAL(KIND=DBLE_PREC), DIMENSION(P) :: PEN,D1PEN,D2PEN,DPENDRHO,ESTIMATE,SUM_X_SQUARES
      REAL(KIND=DBLE_PREC), DIMENSION(N) :: NOISE,WT,Y
      REAL(KIND=DBLE_PREC), DIMENSION(N,P) :: X
!!
!!     Test penalty function
!!
!      RHO = TWO
!      ETA = ONE
!      CALL PENALTY_FUN(BETA,RHO,ETA,"ENET",PEN,D1PEN,D2PEN,DPENDRHO)
!      PRINT*, "BETA="
!      PRINT*, BETA
!      PRINT*, "RHO="
!      PRINT*, RHO
!      PRINT*, "ETA="
!      PRINT*, ETA
!      PRINT*, "PEN="
!      PRINT*, PEN
!      PRINT*, "D1PEN="
!      PRINT*, D1PEN
!      PRINT*, "D2PEN="
!      PRINT*, D2PEN
!      PRINT*, "DPENDRHO="
!      PRINT*, DPENDRHO
!!
!!     Test thresholding function
!!
!      A = ONE
!      B = -ONE
!      RHO = ONE
!      ETA = THREE
!      PRINT*, "A = "
!      PRINT*, A
!      PRINT*, "B = "
!      PRINT*, B
!      PRINT*, "RHO="
!      PRINT*, RHO
!      PRINT*, "ETA="
!      PRINT*, ETA
!      PRINT*, "XMIN = "
!      PRINT*, LSQ_THRESHOLDING(A,B,RHO,ETA,"SCAD")       
!!
!!     Test find max rho function
!!
!      A = ONE
!      B = -ONE
!      PRINT*, "A = "
!      PRINT*, A
!      PRINT*, "B = "
!      PRINT*, B
!      PRINT*, "MAXRHO = ", MAX_RHO(A,B,"MCP",(/FOUR/))
!
!     Test coordinate descent algorithm
!      
      CALL RANDOM_NUMBER(X)
      CALL RANDOM_NUMBER(NOISE)
      Y = MATMUL(X,BETA)+NOISE
      WT = ONE
      PENIDX = .TRUE.
      SUM_X_SQUARES = MATMUL(WT,X**2)
      ESTIMATE = ZERO
      MAXITERS = 0
      LAMBDA = TEN**2
      CALL PENALIZED_L2_REGRESSION(ESTIMATE,X,Y,WT,LAMBDA,&
         SUM_X_SQUARES,PENIDX,MAXITERS,"SCAD",(/FOUR/))
      PRINT*, "ESTIMATE = "
      PRINT*, ESTIMATE
      PAUSE
      END PROGRAM TEST