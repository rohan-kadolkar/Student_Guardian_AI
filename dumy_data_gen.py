import random
import pandas as pd
from datetime import timedelta, datetime
import os

class CompleteDummyDataGenerator:
    def __init__(self, n_students=1000):
        self.n_students = n_students
        self.start_date = datetime(2025, 8, 1)  # Academic year starts Aug 1
        self.current_date = datetime(2025, 12, 1)  # Current date

        # Indian names database
        self.first_names_male = [
            'Aarav', 'Vivaan', 'Aditya', 'Vihaan', 'Arjun', 'Sai', 'Arnav', 'Ayaan',
            'Krishna', 'Ishaan', 'Shaurya', 'Atharv', 'Advait', 'Pranav', 'Dhruv',
            'Aryan', 'Kabir', 'Shivansh', 'Rudra', 'Ritvik', 'Vedant', 'Aadhya',
            'Reyansh', 'Samar', 'Daksh', 'Kiaan', 'Ansh', 'Naksh', 'Yash', 'Kian',
            'Raj', 'Rohan', 'Aditya', 'Amit', 'Rahul', 'Siddharth', 'Varun', 'Kunal',
            'Nikhil', 'Abhinav', 'Karthik', 'Ankit', 'Harshit', 'Manish', 'Prakash'
        ]

        self.first_names_female = [
            'Saanvi', 'Aanya', 'Diya', 'Ananya', 'Pari', 'Aadhya', 'Kavya', 'Sara',
            'Myra', 'Aditi', 'Avni', 'Shanaya', 'Navya', 'Kiara', 'Riya', 'Aarohi',
            'Prisha', 'Anvi', 'Siya', 'Ishita', 'Anika', 'Tara', 'Zara', 'Mira',
            'Ira', 'Nisha', 'Sneha', 'Priya', 'Divya', 'Pooja', 'Anjali', 'Shruti',
            'Meera', 'Tanvi', 'Swati', 'Kriti', 'Simran', 'Neha', 'Ritika', 'Sakshi'
        ]

        self.last_names = [
            'Kumar', 'Singh', 'Sharma', 'Patel', 'Reddy', 'Gupta', 'Mehta', 'Joshi',
            'Desai', 'Agarwal', 'Verma', 'Shah', 'Iyer', 'Nair', 'Pillai', 'Rao',
            'Choudhury', 'Malhotra', 'Kapoor', 'Bose', 'Ghosh', 'Das', 'Chatterjee',
            'Banerjee', 'Mukherjee', 'Jain', 'Sinha', 'Pandey', 'Mishra', 'Tiwari',
            'Saxena', 'Arora', 'Bhatia', 'Khanna', 'Sethi', 'Chopra', 'Kaur', 'Khan',
            'Ali', 'Hussain', 'Ahmed', 'Mohammad', 'Fernandes', "D'Souza", 'Rodrigues'
        ]

        self.cities = [
            'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune',
            'Ahmedabad', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Bhopal',
            'Visakhapatnam', 'Patna', 'Vadodara', 'Ghaziabad', 'Ludhiana', 'Coimbatore'
        ]

        self.subjects = [
            'Mathematics', 'Physics', 'Chemistry', 'Computer Science',
            'Electronics', 'Mechanical Engineering', 'English', 'Data Structures'
        ]

        print(f"📊 Initializing ENHANCED data generator for {n_students} students...")

    def generate_student_master(self):
        """Generate main student data"""
        print("🔄 Generating student master data...")
        students = []

        for i in range(1, self.n_students + 1):
            gender = random.choice(['Male', 'Female'])
            first_name = random.choice(self.first_names_male if gender == 'Male' else self.first_names_female)
            last_name = random.choice(self.last_names)
            age = random.randint(17, 23)
            location = random.choice(self.cities)
            
            location_type = random.choices(['Rural', 'Semi-Urban', 'Urban'], weights=[0.40, 0.35, 0.25], k=1)[0]

            branch_code = random.choice(['CS', 'EC', 'ME', 'CE', 'EE'])
            roll_number = f"{branch_code}{random.randint(21, 24)}{i:04d}"
            email = f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 999)}@student.college.edu"
            phone = f"+91{random.randint(7000000000, 9999999999)}"

            admission_date = self.start_date.strftime('%Y-%m-%d')
            year = random.choice([1, 2, 3, 4])
            semester = random.choice([1, 2]) if year < 4 else 1

            student = {
                'student_id': i,
                'roll_number': roll_number,
                'first_name': first_name,
                'last_name': last_name,
                'full_name': f"{first_name} {last_name}",
                'gender': gender,
                'age': age,
                'date_of_birth': (datetime(2025, 1, 1) - timedelta(days=age*365)).strftime('%Y-%m-%d'),
                'email': email,
                'phone': phone,
                'city': location,
                'location_type': location_type,
                'branch': branch_code,
                'year': year,
                'semester': semester,
                'admission_date': admission_date,
                'status': 'Active'
            }
            students.append(student)

        df = pd.DataFrame(students)
        print(f"✅ Generated {len(df)} student records")
        return df

    def _get_income_range(self, level):
        """Get income range based on level"""
        ranges = {
            'Low': random.randint(50000, 150000),
            'Lower-Middle': random.randint(150000, 300000),
            'Middle': random.randint(300000, 600000),
            'Upper-Middle': random.randint(600000, 1000000),
            'High': random.randint(1000000, 3000000)
        }
        return ranges.get(level, 200000)

    def generate_family_background(self, students_df):
        """Generate family background data"""
        print("🔄 Generating family background data...")
        family_data = []

        for _, student in students_df.iterrows():
            if student['location_type'] == 'Rural':
                income_level = random.choices(['Low', 'Lower-Middle', 'Middle'], weights=[0.60, 0.30, 0.10], k=1)[0]
            elif student['location_type'] == 'Semi-Urban':
                income_level = random.choices(['Low', 'Lower-Middle', 'Middle', 'Upper-Middle'],
                                           weights=[0.25, 0.35, 0.30, 0.10], k=1)[0]
            else:  # Urban
                income_level = random.choices(['Lower-Middle', 'Middle', 'Upper-Middle', 'High'],
                                           weights=[0.20, 0.40, 0.30, 0.10], k=1)[0]

            parent_education_options = ['No Formal Education', 'Primary', 'Secondary',
                                      'Higher Secondary', 'Graduate', 'Post-Graduate']

            father_education = random.choices(parent_education_options,
                                           weights=[0.15, 0.20, 0.25, 0.20, 0.15, 0.05], k=1)[0]
            mother_education = random.choices(parent_education_options,
                                           weights=[0.20, 0.25, 0.25, 0.15, 0.10, 0.05], k=1)[0]

            occupations = ['Farmer', 'Daily Wage Worker', 'Small Business', 'Salaried Employee',
                          'Government Employee', 'Professional', 'Retired', 'Self-Employed']

            father_occupation = random.choice(occupations)
            mother_occupation = random.choices(occupations + ['Homemaker'],
                                            weights=[0.05, 0.05, 0.10, 0.10, 0.08, 0.05, 0.02, 0.10, 0.45], k=1)[0]

            siblings = random.choices([0, 1, 2, 3, 4], weights=[0.15, 0.35, 0.30, 0.15, 0.05], k=1)[0]
            single_parent = random.choices([0, 1], weights=[0.92, 0.08], k=1)[0]

            family = {
                'student_id': student['student_id'],
                'income_level': income_level,
                'annual_income': self._get_income_range(income_level),
                'father_name': f"Mr. {random.choice(self.first_names_male)} {student['last_name']}",
                'father_education': father_education,
                'father_occupation': father_occupation,
                'mother_name': f"Mrs. {random.choice(self.first_names_female)} {student['last_name']}",
                'mother_education': mother_education,
                'mother_occupation': mother_occupation,
                'siblings': siblings,
                'family_size': random.randint(3, 7),
                'single_parent': single_parent,
                'guardian_contact': f"+91{random.randint(7000000000, 9999999999)}",
                'permanent_address': f"{random.randint(1, 500)}, {random.choice(['Street', 'Road', 'Lane'])}, {student['city']}"
            }
            family_data.append(family)

        df = pd.DataFrame(family_data)
        print(f"✅ Generated family background for {len(df)} students")
        return df

    # ============================================
    # 🆕 NEW: ACADEMIC PERFORMANCE HISTORY
    # ============================================
    def generate_academic_history(self, students_df):
        """
        Generate academic history including:
        1. Previous Semester GPA (CRITICAL)
        2. Credit Hours (Registered vs Completed) (CRITICAL)
        3. First Year Performance (CRITICAL)
        4. Course Registration Delays (HIGH)
        5. Mid-Semester Withdrawal History (HIGH)
        """
        print("🔄 Generating academic history data (NEW - CRITICAL FEATURES)...")
        academic_records = []

        for _, student in students_df.iterrows():
            current_year = student['year']
            current_semester = student['semester']
            
            # Calculate total semesters completed
            semesters_completed = (current_year - 1) * 2 + (current_semester - 1)
            
            # Generate performance trend (declining/stable/improving)
            performance_trend = random.choices(
                ['Declining', 'Stable', 'Improving'], 
                weights=[0.25, 0.50, 0.25], 
                k=1
            )[0]
            
            # 1. PREVIOUS SEMESTER GPA (Most important predictor)
            if semesters_completed > 0:
                # Base GPA influenced by year and trend
                if performance_trend == 'Declining':
                    base_gpa = random.uniform(4.0, 7.0)
                    previous_gpa = base_gpa + random.uniform(0.5, 1.5)  # Was higher before
                elif performance_trend == 'Improving':
                    base_gpa = random.uniform(6.0, 9.0)
                    previous_gpa = base_gpa - random.uniform(0.5, 1.5)  # Was lower before
                else:  # Stable
                    base_gpa = random.uniform(5.0, 8.5)
                    previous_gpa = base_gpa + random.uniform(-0.3, 0.3)
                
                current_semester_gpa = round(base_gpa, 2)
                previous_semester_gpa = round(max(0, min(10, previous_gpa)), 2)
            else:
                # First semester students - use entrance exam as proxy
                entrance_percentile = random.uniform(40, 98)
                current_semester_gpa = round((entrance_percentile / 10) * random.uniform(0.8, 1.1), 2)
                previous_semester_gpa = None
            
            # Calculate cumulative GPA
            if semesters_completed > 0:
                cumulative_gpa = round(
                    (previous_semester_gpa * 0.6 + current_semester_gpa * 0.4) 
                    if previous_semester_gpa else current_semester_gpa,
                    2
                )
            else:
                cumulative_gpa = current_semester_gpa
            
            # 2. CREDIT HOURS (Top-3 predictor)
            # Standard credits per semester: 20-24
            credits_registered_this_sem = random.choice([18, 20, 22, 24])
            
            # Calculate completion rate based on performance
            if cumulative_gpa >= 8.0:
                completion_rate = random.uniform(0.95, 1.0)
            elif cumulative_gpa >= 6.0:
                completion_rate = random.uniform(0.80, 0.95)
            elif cumulative_gpa >= 4.0:
                completion_rate = random.uniform(0.60, 0.85)
            else:
                completion_rate = random.uniform(0.40, 0.70)
            
            credits_completed_this_sem = int(credits_registered_this_sem * completion_rate)
            
            # Total credits (cumulative)
            total_credits_registered = credits_registered_this_sem * (semesters_completed + 1)
            total_credits_completed = int(total_credits_registered * completion_rate)
            
            # Credit completion rate (CRITICAL METRIC)
            credit_completion_rate = round((total_credits_completed / total_credits_registered * 100), 2)
            
            # 3. FIRST YEAR PERFORMANCE (Year 1 students at highest risk)
            if current_year == 1:
                first_year_gpa = current_semester_gpa
                first_year_credits_completed = credits_completed_this_sem
                first_year_attendance = random.uniform(60, 95)  # Will correlate with attendance data
                first_year_dropout_risk = 'High' if first_year_gpa < 5.0 else 'Medium' if first_year_gpa < 7.0 else 'Low'
            else:
                # Past first year performance
                first_year_gpa = round(cumulative_gpa + random.uniform(-1.0, 0.5), 2)
                first_year_credits_completed = random.choice([36, 38, 40, 42, 44])  # 2 semesters
                first_year_attendance = random.uniform(65, 95)
                first_year_dropout_risk = 'Low'  # They survived first year
            
            # 4. COURSE REGISTRATION DELAYS (Indicator of disengagement)
            # Expected registration date: 5 days before semester start
            expected_registration_date = self.start_date - timedelta(days=5)
            
            # Some students register late
            registration_delay_days = random.choices(
                [0, 1, 2, 3, 5, 7, 10, 15, 20],
                weights=[0.50, 0.15, 0.10, 0.08, 0.07, 0.05, 0.03, 0.01, 0.01],
                k=1
            )[0]
            
            actual_registration_date = expected_registration_date + timedelta(days=registration_delay_days)
            registration_status = 'On Time' if registration_delay_days <= 2 else 'Late' if registration_delay_days <= 7 else 'Very Late'
            
            # 5. MID-SEMESTER WITHDRAWAL HISTORY (Strong predictor)
            # Number of courses dropped in previous semesters
            if semesters_completed > 0:
                # Higher chance of withdrawals for low performers
                if cumulative_gpa < 5.0:
                    courses_withdrawn_ever = random.choices([0, 1, 2, 3], weights=[0.30, 0.40, 0.20, 0.10], k=1)[0]
                elif cumulative_gpa < 7.0:
                    courses_withdrawn_ever = random.choices([0, 1, 2], weights=[0.60, 0.30, 0.10], k=1)[0]
                else:
                    courses_withdrawn_ever = random.choices([0, 1], weights=[0.85, 0.15], k=1)[0]
            else:
                courses_withdrawn_ever = 0
            
            # Withdrawals in current semester
            courses_withdrawn_current = 1 if random.random() < 0.05 else 0  # 5% chance
            
            # Total withdrawals
            total_withdrawals = courses_withdrawn_ever + courses_withdrawn_current
            
            # Calculate semester number
            semester_number = semesters_completed + 1
            
            # BONUS: Course failures
            if cumulative_gpa < 5.0:
                courses_failed = random.choices([0, 1, 2, 3, 4], weights=[0.10, 0.30, 0.30, 0.20, 0.10], k=1)[0]
            elif cumulative_gpa < 7.0:
                courses_failed = random.choices([0, 1, 2], weights=[0.50, 0.35, 0.15], k=1)[0]
            else:
                courses_failed = random.choices([0, 1], weights=[0.90, 0.10], k=1)[0]
            
            # BONUS: Course repeats
            courses_repeated = min(courses_failed, random.randint(0, 2))
            
            academic_record = {
                'student_id': student['student_id'],
                
                # 1. GPA METRICS (CRITICAL - #1 Predictor)
                'current_semester_gpa': current_semester_gpa,
                'previous_semester_gpa': previous_semester_gpa,
                'cumulative_gpa': cumulative_gpa,
                'gpa_trend': performance_trend,
                
                # 2. CREDIT HOURS (CRITICAL - Top-3 Predictor)
                'credits_registered_current': credits_registered_this_sem,
                'credits_completed_current': credits_completed_this_sem,
                'total_credits_registered': total_credits_registered,
                'total_credits_completed': total_credits_completed,
                'credit_completion_rate': credit_completion_rate,
                
                # 3. FIRST YEAR PERFORMANCE (CRITICAL)
                'first_year_gpa': round(first_year_gpa, 2),
                'first_year_credits_completed': first_year_credits_completed,
                'first_year_attendance_percent': round(first_year_attendance, 2),
                'first_year_dropout_risk': first_year_dropout_risk,
                
                # 4. REGISTRATION (HIGH Impact)
                'registration_date': actual_registration_date.strftime('%Y-%m-%d'),
                'registration_delay_days': registration_delay_days,
                'registration_status': registration_status,
                
                # 5. WITHDRAWALS (HIGH Impact)
                'courses_withdrawn_ever': courses_withdrawn_ever,
                'courses_withdrawn_current': courses_withdrawn_current,
                'total_course_withdrawals': total_withdrawals,
                
                # BONUS METRICS
                'semester_number': semester_number,
                'courses_failed_ever': courses_failed,
                'courses_repeated': courses_repeated,
                'academic_standing': self._calculate_academic_standing(cumulative_gpa),
                'probation_status': 'Yes' if cumulative_gpa < 5.0 else 'No'
            }
            
            academic_records.append(academic_record)

        df = pd.DataFrame(academic_records)
        print(f"✅ Generated academic history for {len(df)} students with {len(df.columns)} critical features")
        return df
    
    def _calculate_academic_standing(self, gpa):
        """Calculate academic standing based on GPA"""
        if gpa >= 8.5:
            return 'Excellent'
        elif gpa >= 7.0:
            return 'Good'
        elif gpa >= 5.5:
            return 'Average'
        elif gpa >= 4.0:
            return 'Below Average'
        else:
            return 'Poor'

    # ============================================
    # EXISTING METHODS (UNCHANGED)
    # ============================================
    
    def generate_daily_attendance(self, students_df):
        """Generate daily attendance for past 8 weeks"""
        print("🔄 Generating daily attendance records (8 weeks)...")
        attendance_records = []
        weeks = 8

        for _, student in students_df.iterrows():
            base_rate = random.uniform(0.65, 0.95)
            current_date_iter = self.current_date - timedelta(weeks=weeks)

            for week in range(weeks):
                for day in range(5):  # Mon-Fri
                    date = current_date_iter + timedelta(days=week*7 + day)
                    if date > self.current_date:
                        continue

                    day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][day]
                    if day == 0: prob = base_rate * 0.85
                    elif day == 4: prob = base_rate * 0.90
                    else: prob = base_rate

                    if random.random() < 0.25:
                        prob -= (week / weeks) * 0.2

                    status = 'Present' if random.random() < prob else 'Absent'

                    record = {
                        'student_id': student['student_id'],
                        'date': date.strftime('%Y-%m-%d'),
                        'day_of_week': day_name,
                        'week_number': week + 1,
                        'status': status,
                        'marked_by': random.choice(['Prof. Sharma', 'Prof. Mehta', 'Prof. Patel']),
                        'marked_at': date.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    attendance_records.append(record)

        df = pd.DataFrame(attendance_records)
        print(f"✅ Generated {len(df)} attendance records")
        return df

    def generate_marks_data(self, students_df):
        """Generate marks for different exams/tests"""
        print("🔄 Generating marks data...")
        marks_records = []
        exam_types = [
            ('Unit Test 1', 20, '2025-09-15'),
            ('Unit Test 2', 20, '2025-10-20'),
            ('Mid Semester', 30, '2025-11-10'),
            ('Assignment 1', 10, '2025-09-30'),
            ('Assignment 2', 10, '2025-10-30'),
            ('Lab Internal', 10, '2025-11-20')
        ]

        for _, student in students_df.iterrows():
            aptitude = random.uniform(40, 95)
            for subject in self.subjects[:6]:
                for exam_name, max_marks, exam_date in exam_types:
                    performance_factor = random.uniform(0.7, 1.1)
                    obtained = aptitude * (max_marks / 100) * performance_factor
                    obtained = max(0, min(max_marks, obtained + random.uniform(-3, 3)))
                    
                    if random.random() < 0.20:
                        if 'Test 2' in exam_name or 'Mid' in exam_name:
                            obtained *= 0.75

                    record = {
                        'student_id': student['student_id'],
                        'subject': subject,
                        'exam_type': exam_name,
                        'max_marks': max_marks,
                        'obtained_marks': round(obtained, 2),
                        'percentage': round((obtained / max_marks) * 100, 2),
                        'exam_date': exam_date,
                        'evaluator': random.choice(['Prof. Kumar', 'Prof. Singh', 'Prof. Gupta'])
                    }
                    marks_records.append(record)

        df = pd.DataFrame(marks_records)
        print(f"✅ Generated {len(df)} marks records")
        return df

    def generate_assignments(self, students_df):
        print("🔄 Generating assignment submission data...")
        assignment_records = []
        assignments = [
            ('Assignment 1 - Subject A', '2025-09-15', '2025-09-30'),
            ('Assignment 2 - Subject B', '2025-10-01', '2025-10-15'),
            ('Assignment 3 - Subject C', '2025-10-20', '2025-11-05'),
            ('Assignment 4 - Subject D', '2025-11-10', '2025-11-25'),
            ('Project Proposal', '2025-09-20', '2025-10-10'),
            ('Lab Report 1', '2025-10-05', '2025-10-20'),
            ('Lab Report 2', '2025-11-01', '2025-11-15'),
            ('Case Study', '2025-11-15', '2025-11-30')
        ]

        for _, student in students_df.iterrows():
            for assignment_name, assigned_date, due_date in assignments:
                submit_prob = random.uniform(0.60, 0.95)
                submitted = random.random() < submit_prob

                if submitted:
                    due_dt = datetime.strptime(due_date, '%Y-%m-%d')
                    if random.random() < 0.75:
                        submit_date = due_dt - timedelta(days=random.randint(0, 3))
                    else:
                        submit_date = due_dt + timedelta(days=random.randint(1, 7))

                    submitted_on = submit_date.strftime('%Y-%m-%d')
                    status = 'On Time' if submit_date <= due_dt else 'Late'
                    grade = random.choices(['A', 'B', 'C', 'D'], weights=[0.30, 0.40, 0.20, 0.10], k=1)[0]
                else:
                    submitted_on = None
                    status = 'Not Submitted'
                    grade = None

                record = {
                    'student_id': student['student_id'],
                    'assignment_name': assignment_name,
                    'assigned_date': assigned_date,
                    'due_date': due_date,
                    'submitted': 'Yes' if submitted else 'No',
                    'submission_date': submitted_on,
                    'status': status,
                    'grade': grade,
                    'feedback': 'Good work' if grade in ['A', 'B'] else 'Needs improvement' if grade else None
                }
                assignment_records.append(record)

        df = pd.DataFrame(assignment_records)
        print(f"✅ Generated {len(df)} assignment records")
        return df

    def generate_behavior_reports(self, students_df):
        print("🔄 Generating behavior reports...")
        behavior_records = []
        behaviors = [
            ('Excellent Participation', 'Positive', 'Always participates actively in class'),
            ('Punctual', 'Positive', 'Always on time for classes'),
            ('Helpful to Peers', 'Positive', 'Assists other students'),
            ('Late to Class', 'Negative', 'Frequently arrives late'),
            ('Irregular Attendance', 'Negative', 'Missing classes without notice'),
            ('Distracted in Class', 'Negative', 'Not paying attention'),
            ('Good Team Player', 'Positive', 'Works well in groups'),
            ('Respectful Behavior', 'Positive', 'Shows respect to faculty and peers')
        ]

        students_with_reports = students_df.sample(frac=0.30)

        for _, student in students_with_reports.iterrows():
            num_reports = random.randint(1, 3)
            for _ in range(num_reports):
                behavior, type_, description = random.choice(behaviors)
                report_date = (self.current_date - timedelta(days=random.randint(1, 90))).strftime('%Y-%m-%d')

                record = {
                    'student_id': student['student_id'],
                    'report_date': report_date,
                    'behavior_type': type_,
                    'behavior': behavior,
                    'description': description,
                    'reported_by': random.choice(['Prof. Sharma', 'Prof. Mehta', 'Prof. Reddy', 'HOD']),
                    'severity': random.choice(['Low', 'Medium', 'High']) if type_ == 'Negative' else 'Positive'
                }
                behavior_records.append(record)

        df = pd.DataFrame(behavior_records)
        print(f"✅ Generated {len(df)} behavior reports")
        return df

    def generate_library_usage(self, students_df):
        print("🔄 Generating library usage data...")
        library_records = []
        books = [
            'Introduction to Algorithms', 'Data Structures and Algorithms',
            'Computer Networks', 'Operating Systems', 'Database Management Systems',
            'Software Engineering', 'Artificial Intelligence', 'Machine Learning',
            'Physics Vol 1', 'Chemistry Fundamentals', 'Advanced Mathematics',
            'Digital Electronics', 'Control Systems', 'Power Systems'
        ]

        active_library_users = students_df.sample(frac=0.60)

        for _, student in active_library_users.iterrows():
            num_visits = random.randint(5, 20)
            for _ in range(num_visits):
                visit_date = (self.current_date - timedelta(days=random.randint(1, 100))).strftime('%Y-%m-%d')
                activity = random.choice(['Book Borrowed', 'Reading Session', 'Book Returned'])
                
                record = {
                    'student_id': student['student_id'],
                    'visit_date': visit_date,
                    'activity': activity,
                    'book_title': random.choice(books) if 'Book' in activity else None,
                    'duration_hours': random.randint(1, 4) if activity == 'Reading Session' else None,
                    'checkout_date': visit_date if activity == 'Book Borrowed' else None,
                    'return_date': (datetime.strptime(visit_date, '%Y-%m-%d') + timedelta(days=random.randint(7, 21))).strftime('%Y-%m-%d') if activity == 'Book Returned' else None
                }
                library_records.append(record)

        df = pd.DataFrame(library_records)
        print(f"✅ Generated {len(df)} library usage records")
        return df

    def generate_fee_payments(self, students_df, family_df):
        print("🔄 Generating fee payment data...")
        fee_records = []
        merged = students_df.merge(family_df[['student_id', 'income_level']], on='student_id')

        installments = [
            ('Semester 1 - Tuition', 35000, '2025-08-15'),
            ('Semester 1 - Other', 15000, '2025-08-15'),
            ('Semester 2 - Tuition', 35000, '2025-01-15'),
            ('Semester 2 - Other', 15000, '2025-01-15')
        ]

        for _, student in merged.iterrows():
            if student['income_level'] == 'Low':
                payment_prob, delay_prob = 0.60, 0.50
            elif student['income_level'] in ['Lower-Middle', 'Middle']:
                payment_prob, delay_prob = 0.85, 0.30
            else:
                payment_prob, delay_prob = 0.98, 0.10

            total_paid = 0

            for installment_name, amount, due_date in installments:
                due_dt = datetime.strptime(due_date, '%Y-%m-%d')
                if due_dt > self.current_date:
                    continue

                paid = random.random() < payment_prob
                if paid:
                    if random.random() < delay_prob:
                        payment_date = due_dt + timedelta(days=random.randint(10, 60))
                    else:
                        payment_date = due_dt - timedelta(days=random.randint(0, 10))
                    
                    status = 'Paid - On Time' if payment_date <= due_dt else 'Paid - Late'
                    paid_amount = amount
                    total_paid += amount
                else:
                    payment_date = None
                    status = 'Pending'
                    paid_amount = 0

                record = {
                    'student_id': student['student_id'],
                    'installment': installment_name,
                    'amount_due': amount,
                    'due_date': due_date,
                    'payment_date': payment_date.strftime('%Y-%m-%d') if payment_date else None,
                    'amount_paid': paid_amount,
                    'status': status,
                    'payment_method': random.choice(['Online', 'Cheque', 'Cash', 'DD']) if paid else None,
                    'receipt_number': f"RCP{random.randint(10000, 99999)}" if paid else None
                }
                fee_records.append(record)

        df = pd.DataFrame(fee_records)
        print(f"✅ Generated {len(df)} fee payment records")
        return df

    def save_all_data(self, output_dir='dummy_data'):
        """Generate and save all data to CSV files"""
        print("\n" + "="*80)
        print("GENERATING COMPLETE DUMMY DATA FOR 1000+ STUDENTS")
        print("🆕 NOW INCLUDING CRITICAL MISSING FEATURES!")
        print("="*80 + "\n")

        os.makedirs(output_dir, exist_ok=True)

        students_df = self.generate_student_master()
        students_df.to_csv(f'{output_dir}/01_students_master.csv', index=False)

        family_df = self.generate_family_background(students_df)
        family_df.to_csv(f'{output_dir}/02_family_background.csv', index=False)
        
        # 🆕 NEW: Academic History (CRITICAL FEATURES)
        academic_df = self.generate_academic_history(students_df)
        academic_df.to_csv(f'{output_dir}/03_academic_history.csv', index=False)

        attendance_df = self.generate_daily_attendance(students_df)
        attendance_df.to_csv(f'{output_dir}/04_daily_attendance.csv', index=False)

        marks_df = self.generate_marks_data(students_df)
        marks_df.to_csv(f'{output_dir}/05_marks_exams.csv', index=False)

        assignments_df = self.generate_assignments(students_df)
        assignments_df.to_csv(f'{output_dir}/06_assignments.csv', index=False)

        behavior_df = self.generate_behavior_reports(students_df)
        behavior_df.to_csv(f'{output_dir}/07_behavior_reports.csv', index=False)

        library_df = self.generate_library_usage(students_df)
        library_df.to_csv(f'{output_dir}/08_library_usage.csv', index=False)

        fee_df = self.generate_fee_payments(students_df, family_df)
        fee_df.to_csv(f'{output_dir}/09_fee_payments.csv', index=False)

        self._generate_summary_report(output_dir, students_df, family_df, academic_df, attendance_df,
                                    marks_df, assignments_df, behavior_df, library_df, fee_df)

        print("\n" + "="*80)
        print("✅ ALL DATA GENERATED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 Output Directory: {output_dir}/")
        print(f"\n📊 Generated Files:")
        print(f"   1. 01_students_master.csv       - {len(students_df)} students")
        print(f"   2. 02_family_background.csv     - {len(family_df)} records")
        print(f"   3. 🆕 03_academic_history.csv   - {len(academic_df)} records ⭐ NEW!")
        print(f"   4. 04_daily_attendance.csv      - {len(attendance_df)} records")
        print(f"   5. 05_marks_exams.csv           - {len(marks_df)} records")
        print(f"   6. 06_assignments.csv           - {len(assignments_df)} records")
        print(f"   7. 07_behavior_reports.csv      - {len(behavior_df)} records")
        print(f"   8. 08_library_usage.csv         - {len(library_df)} records")
        print(f"   9. 09_fee_payments.csv          - {len(fee_df)} records")
        print(f"  10. 00_data_summary.txt          - Overview report")
        
        return {
            'students': students_df,
            'family': family_df,
            'academic_history': academic_df,  # NEW
            'attendance': attendance_df,
            'marks': marks_df,
            'assignments': assignments_df,
            'behavior': behavior_df,
            'library': library_df,
            'fee': fee_df
        }

    def _generate_summary_report(self, output_dir, students_df, family_df, academic_df, attendance_df,
                                marks_df, assignments_df, behavior_df, library_df, fee_df):
        """Generate summary statistics"""

        attendance_grouped = attendance_df.groupby('student_id')['status'].apply(
            lambda x: (x == 'Present').sum() / len(x) * 100
        )
        avg_attendance = attendance_grouped.mean()

        marks_grouped = marks_df.groupby('student_id')['percentage'].mean()
        avg_marks = marks_grouped.mean()

        assignments_grouped = assignments_df.groupby('student_id')['submitted'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        )
        avg_assignment_completion = assignments_grouped.mean()

        fee_grouped = fee_df.groupby('student_id')['amount_paid'].sum()
        total_fee_collected = fee_grouped.sum()

        summary = f"""
{'='*80}
DUMMY DATA GENERATION SUMMARY (ENHANCED VERSION)
{'='*80}

Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Students: {len(students_df)}

## DEMOGRAPHICS

Male Students: {(students_df['gender'] == 'Male').sum()}
Female Students: {(students_df['gender'] == 'Female').sum()}

Location Distribution:
- Rural: {(students_df['location_type'] == 'Rural').sum()}
- Semi-Urban: {(students_df['location_type'] == 'Semi-Urban').sum()}
- Urban: {(students_df['location_type'] == 'Urban').sum()}

## FAMILY BACKGROUND

Income Level Distribution:
- Low: {(family_df['income_level'] == 'Low').sum()}
- Lower-Middle: {(family_df['income_level'] == 'Lower-Middle').sum()}
- Middle: {(family_df['income_level'] == 'Middle').sum()}
- Upper-Middle: {(family_df['income_level'] == 'Upper-Middle').sum()}
- High: {(family_df['income_level'] == 'High').sum()}

## 🆕 ACADEMIC HISTORY (NEW - CRITICAL FEATURES)

Average Cumulative GPA: {academic_df['cumulative_gpa'].mean():.2f}
Average Credit Completion Rate: {academic_df['credit_completion_rate'].mean():.2f}%

Students on Academic Probation: {(academic_df['probation_status'] == 'Yes').sum()}
Students with Course Withdrawals: {(academic_df['total_course_withdrawals'] > 0).sum()}
Late Registration Students: {(academic_df['registration_status'] != 'On Time').sum()}

First Year Students at High Risk: {(academic_df['first_year_dropout_risk'] == 'High').sum()}

GPA Distribution:
- Excellent (8.5+): {(academic_df['cumulative_gpa'] >= 8.5).sum()}
- Good (7.0-8.5): {((academic_df['cumulative_gpa'] >= 7.0) & (academic_df['cumulative_gpa'] < 8.5)).sum()}
- Average (5.5-7.0): {((academic_df['cumulative_gpa'] >= 5.5) & (academic_df['cumulative_gpa'] < 7.0)).sum()}
- Below Average (4.0-5.5): {((academic_df['cumulative_gpa'] >= 4.0) & (academic_df['cumulative_gpa'] < 5.5)).sum()}
- Poor (<4.0): {(academic_df['cumulative_gpa'] < 4.0).sum()}

## ATTENDANCE

Total Attendance Records: {len(attendance_df)}
Average Attendance: {avg_attendance:.2f}%
Students with <75% Attendance: {(attendance_grouped < 75).sum()}

## MARKS & EXAMS

Total Marks Records: {len(marks_df)}
Average Marks: {avg_marks:.2f}%
Students Scoring <40%: {(marks_grouped < 40).sum()}

## ASSIGNMENTS

Total Assignment Records: {len(assignments_df)}
Average Completion Rate: {avg_assignment_completion:.2f}%
Students with <50% Completion: {(assignments_grouped < 50).sum()}

## BEHAVIOR REPORTS

Total Behavior Reports: {len(behavior_df)}
Positive Reports: {(behavior_df['behavior_type'] == 'Positive').sum()}
Negative Reports: {(behavior_df['behavior_type'] == 'Negative').sum()}

## LIBRARY USAGE

Total Library Records: {len(library_df)}
Active Library Users: {library_df['student_id'].nunique()}
Average Visits per Student: {len(library_df) / library_df['student_id'].nunique():.1f}

## FEE PAYMENTS

Total Fee Records: {len(fee_df)}
Total Fee Collected: ₹{total_fee_collected:,.2f}
Students with Pending Fees: {(fee_df[fee_df['status'] == 'Pending'].groupby('student_id').size() > 0).sum()}

{'='*80}
🆕 CRITICAL FEATURES ADDED:
1. ✅ Previous Semester GPA
2. ✅ Credit Hours (Completed vs Registered)
3. ✅ First Year Performance
4. ✅ Course Registration Delays
5. ✅ Mid-Semester Withdrawal History
{'='*80}
"""
        
        with open(f'{output_dir}/00_data_summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary)

        print(summary)


if __name__ == "__main__":
    generator = CompleteDummyDataGenerator(n_students=1000)
    generator.save_all_data()
    
    print("\n🎉 DONE! Enhanced dataset with CRITICAL features generated!")
    print("📂 Check the 'dummy_data' folder for all CSV files.")
    print("\n💡 You now have:")
    print("   ✅ All previous data (8 CSV files)")
    print("   🆕 Academic history with 5 CRITICAL dropout predictors")
    print("   📊 Ready for high-accuracy ML model training!")